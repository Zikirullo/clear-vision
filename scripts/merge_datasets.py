"""Merge every raw dataset into one YOLO dataset with the classes in configs/classes.yaml.

Each source has a mapping file configs/mappings/<short_name>.yaml:

    folder: rf_cone              # folder under a --raw folder
    map: {cone: traffic_cone}    # source class -> target class, or "drop"
    review: [some-class]         # optional: save sample images with these boxes drawn
    unlisted: drop               # optional: drop classes missing from `map` (default: fail)
    missing: warn                # optional: mapped classes the source lacks only warn (default: fail)
    format: mapillary            # optional: instance masks instead of YOLO labels

The result is ONE file on Drive, not a folder of ~100k files. Drive's background upload
lost files silently when the folder didn't fit:

    datasets/merged.zip          # ZIP_STORED: data.yaml, images/, labels/, report, md5 index
    datasets/merge_report.json   # boxes per class per split, per source
    datasets/review/<class>/     # sample images of classes marked for review
    datasets/merged.done         # written only after the upload was verified

Full merge:
1. Scan and validate every source. Nothing is copied if a source class isn't mapped.
2. Copy the images to the local work dir. Images over --max-side px become JPEGs of
   that size (q90), and PNGs become JPEGs. Exact duplicates (md5) are dropped.
3. Grouped train/val/test split. Roboflow augmented copies stay together, and each
   dataset's own test split stays in test.
4. Write labels, data.yaml, the report and the md5 index, then zip.
5. Check free Drive space and write the zip. Flush Drive (unmount + mount) and verify
   the zip's size and CRCs, then write merged.done.

Append (--append NAME ...):
- Copies merged.zip to the local disk, adds the new sources (same rules, de-duplicated
  against merged images, split on their own), and re-maps merged sources whose
  mapping file changed.
- Then re-zips and uploads with the same space check and verification.

Training notebooks load it with src/colab_data.py prepare_dataset().

Usage (Colab, see notebooks/02_merge_datasets.ipynb):
    python scripts/merge_datasets.py \\
        --raw /content/raw /content/drive/MyDrive/clear_vision/datasets/raw \\
        --zip /content/drive/MyDrive/clear_vision/datasets/merged.zip \\
        --work /content/merged_tmp
    python scripts/merge_datasets.py --append rf_new_source --raw ... --zip ... --work ...
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

logger = logging.getLogger("merge")

REPO = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DROP = "drop"
WORK_MARKER = ".merge_work"  # only folders carrying this marker are ever deleted
REPORT_FILE = "merge_report.json"
INDEX_FILE = "md5_index.json"  # md5 of each merged image's source file -> "split/name"
LOCAL_DATASET = "/content/datasets/merged"  # data.yaml `path`; prepare_dataset() rewrites it
DEFAULT_WORK = Path("/content/merged_tmp")
MIN_TRAIN_BOXES = 300
MAX_MISSING_LABELS = 0.5  # fail if more than this share of a source's images lack a label file
SPACE_MARGIN = 1.1  # need 10% more free space than the files take
WORKERS = 16  # threads for file I/O; Drive reads are slow but parallelize well
COPY_CHUNK = 16 * 1024 * 1024

Box = tuple[int, float, float, float, float]  # class id, x_center, y_center, w, h (0-1)
ReviewBox = tuple[str, float, float, float, float]  # source class name + box
RawArg = Path | str | Sequence[Path | str]
SyncFn = Callable[[], None]


class MergeError(Exception):
    """A problem that would make the merged dataset wrong. The merge stops."""


@dataclass(frozen=True)
class Mapping:
    short_name: str
    folder: str
    classes: dict[str, str]  # source class -> target class or DROP
    review: tuple[str, ...] = ()
    unlisted_drop: bool = False
    missing_warn: bool = False
    format: str = "yolo"  # "yolo" or "mapillary"
    min_size_px: int = 12  # mapillary only


@dataclass
class Record:
    source: str
    image: Path
    group: str  # images of one group always land in the same split
    own_test: bool
    boxes: list[Box]
    review: list[ReviewBox] = field(default_factory=list)
    out_stem: str = ""
    md5: str = ""  # of the source file
    split: str = ""

    @property
    def out_name(self) -> str:
        return self.out_stem + ".jpg"  # every merged image is a JPEG


@dataclass
class SourceStats:
    folder: str
    images: int = 0
    duplicates_removed: int = 0
    missing_label_files: int = 0  # kept as background images
    invalid_boxes: int = 0
    small_instances_skipped: int = 0
    dropped_boxes: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class ZipTarget:
    """merged.zip on Drive and the files next to it."""

    zip: Path

    @property
    def report(self) -> Path:
        return self.zip.with_name(REPORT_FILE)

    @property
    def review(self) -> Path:
        return self.zip.parent / "review"

    @property
    def done(self) -> Path:
        return self.zip.with_suffix(".done")


# ---------------------------------------------------------------------- config


def as_roots(raw: RawArg) -> list[Path]:
    """One or more raw folders, searched in order for each source folder."""
    return [Path(raw)] if isinstance(raw, (str, Path)) else [Path(r) for r in raw]


def find_source_dir(roots: Sequence[Path], folder: str) -> Path | None:
    for root in roots:
        if (root / folder).is_dir():
            return root / folder
    return None


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(Path(path).read_text()) or {}).get("names")
    if not names or len(set(names)) != len(names):
        raise MergeError(f"{path}: needs a `names` list without duplicates")
    return [str(n) for n in names]


def load_mappings(mapping_dir: Path, classes: Sequence[str]) -> list[Mapping]:
    files = sorted(Path(mapping_dir).glob("*.yaml"))
    if not files:
        raise MergeError(f"No mapping files in {mapping_dir}")
    mappings = []
    for path in files:
        data = yaml.safe_load(path.read_text()) or {}
        # str(): YAML reads an unquoted class name like 2 as a number.
        cmap = {str(k): str(v) for k, v in (data.get("map") or {}).items()}
        if not cmap:
            raise MergeError(f"{path.name}: empty `map`")
        unknown = sorted({v for v in cmap.values() if v != DROP and v not in classes})
        if unknown:
            raise MergeError(
                f"{path.name}: target classes {unknown} are not in classes.yaml. "
                f"Use one of {list(classes)} or '{DROP}'."
            )
        review = tuple(str(c) for c in data.get("review") or [])
        not_mapped = [c for c in review if c not in cmap]
        if not_mapped:
            raise MergeError(f"{path.name}: review classes {not_mapped} are not in `map`")
        mappings.append(
            Mapping(
                short_name=path.stem,
                folder=str(data.get("folder", path.stem)),
                classes=cmap,
                review=review,
                unlisted_drop=data.get("unlisted") == DROP,
                missing_warn=data.get("missing") == "warn",
                format=str(data.get("format", "yolo")),
                min_size_px=int(data.get("min_size_px", 12)),
            )
        )
    return mappings


def active_sources(roots: Sequence[Path], mappings: Sequence[Mapping]) -> list[Mapping]:
    """Mappings whose folder exists. Fails on missing sources and unmapped folders."""
    active = []
    for m in mappings:
        if find_source_dir(roots, m.folder):
            active.append(m)
        elif m.format == "mapillary":
            logger.info("%s not found: %s skipped", m.folder, m.short_name)
        else:
            raise MergeError(
                f"{m.short_name}: folder {m.folder} not found in {[str(r) for r in roots]}. "
                f"Download it (notebook 02 section 2 puts new sources in /content/raw), or "
                f"delete configs/mappings/{m.short_name}.yaml."
            )
    covered = {Path(m.folder).parts[0] for m in mappings}
    for root in roots:
        unmapped = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in covered
        )
        if unmapped:
            raise MergeError(
                f"Folders in {root} without a mapping file: {unmapped}. Add "
                "configs/mappings/<name>.yaml for each, so nothing is left out silently."
            )
    return active


def log_mappings(mappings: Sequence[Mapping], active: Sequence[Mapping]) -> None:
    logger.info("Class mappings (source class -> target class):")
    for m in mappings:
        present = "" if m in active else "   (not present, skipped)"
        logger.info("  %s   [%s]%s", m.short_name, m.folder, present)
        for src, dst in m.classes.items():
            logger.info("      %-40s -> %s", src, "DROP" if dst == DROP else dst)
        if m.unlisted_drop:
            logger.info("      %-40s -> DROP", "(any other class)")
        for cls in m.review:
            logger.info("      review samples of %r will be saved", cls)
    logger.info("")


def check_classes(m: Mapping, names: Sequence[str], where: str) -> None:
    """Every source class must be mapped, and every mapped class must exist."""
    unmapped = [n for n in names if n not in m.classes]
    if unmapped and not m.unlisted_drop:
        raise MergeError(
            f"{m.short_name}: classes {unmapped} from {where} are not in "
            f"configs/mappings/{m.short_name}.yaml. Map each one to a target class or to "
            f"'{DROP}', so no boxes are lost silently."
        )
    if unmapped:
        logger.info("%s: dropping unlisted classes %s", m.short_name, unmapped)
    for missing in (c for c in m.classes if c not in names):
        close = difflib.get_close_matches(missing, names, n=5, cutoff=0.4)
        msg = (
            f"{m.short_name}: mapping lists class {missing!r}, but {where} has no such "
            f"class. Similar names: {close or 'none'}."
        )
        if not m.missing_warn:
            raise MergeError(msg)
        logger.warning("WARNING %s", msg)


# ---------------------------------------------------------------------- scanning


def group_key(stem: str) -> str:
    """Roboflow augmented copies are named <original>.rf.<hash>, so group by <original>."""
    return stem.split(".rf.")[0]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def find_data_yaml(root: Path) -> Path:
    for name in ("data.yaml", "dataset.yaml"):
        if (root / name).is_file():
            return root / name
    found = sorted(root.rglob("data.yaml")) + sorted(root.rglob("dataset.yaml"))
    if not found:
        raise MergeError(f"No data.yaml or dataset.yaml in {root}")
    return found[0]


def read_names(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text()) or {}).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        raise MergeError(f"{path}: no class `names`")
    return [str(n) for n in names]


def label_path(image: Path, root: Path) -> Path:
    """Ultralytics convention: the last `images` folder becomes `labels`."""
    parts = list(image.relative_to(root).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return root.joinpath(*parts).with_suffix(".txt")


def is_own_test(image: Path, root: Path) -> bool:
    return any(p.lower() in ("test", "testing") for p in image.relative_to(root).parts[:-1])


def parse_label_text(text: str, n_classes: int, where: Path) -> tuple[list[Box], int]:
    """Parse YOLO boxes or polygons (converted to boxes). Returns (boxes, invalid lines)."""
    boxes: list[Box] = []
    invalid = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            invalid += 1
            continue
        if not 0 <= cls < n_classes:
            raise MergeError(f"{where}: class id {cls} is outside the {n_classes} names in its data.yaml")
        if len(values) == 4:
            x, y, w, h = values
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            invalid += 1
            continue
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 <= x1 or y2 <= y1:
            invalid += 1
            continue
        boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes, invalid


def label_text(boxes: Iterable[Box]) -> str:
    return "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes)


def map_boxes(
    m: Mapping, names: Sequence[str], src_boxes: Iterable[Box], target_ids: dict[str, int],
    stats: SourceStats,
) -> tuple[list[Box], list[ReviewBox]]:
    boxes: list[Box] = []
    review: list[ReviewBox] = []
    for cls, x, y, w, h in src_boxes:
        name = names[cls]
        if name in m.review:
            review.append((name, x, y, w, h))
        target = m.classes.get(name, DROP)
        if target == DROP:
            stats.dropped_boxes[name] += 1
        else:
            boxes.append((target_ids[target], x, y, w, h))
    return boxes, review


def threaded(fn: Callable[[Any], Any], items: Sequence[Any], desc: str) -> list[Any]:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))


def scan_yolo(m: Mapping, root: Path, target_ids: dict[str, int], stats: SourceStats) -> list[Record]:
    yaml_path = find_data_yaml(root)
    names = read_names(yaml_path)
    check_classes(m, names, f"{m.folder}/{yaml_path.relative_to(root)}")
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not images:
        raise MergeError(f"{m.short_name}: no images in {root}")

    def load(image: Path) -> tuple[list[Box], int, bool]:
        path = label_path(image, root)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return [], 0, True
        boxes, invalid = parse_label_text(text, len(names), path)
        return boxes, invalid, False

    records = []
    for image, (src_boxes, invalid, missing) in zip(images, threaded(load, images, f"scan {m.short_name}")):
        stats.invalid_boxes += invalid
        stats.missing_label_files += missing
        boxes, review = map_boxes(m, names, src_boxes, target_ids, stats)
        records.append(Record(
            source=m.short_name,
            image=image,
            group=f"{m.short_name}/{group_key(image.stem)}",
            own_test=is_own_test(image, root),
            boxes=boxes,
            review=review,
        ))
    # A few background images are normal; mostly missing labels means a layout we don't know.
    if stats.missing_label_files > len(images) * MAX_MISSING_LABELS:
        example = images[0]
        raise MergeError(
            f"{m.short_name}: {stats.missing_label_files} of {len(images)} images have no label "
            f"file. For {example.relative_to(root)} the label was expected at "
            f"{label_path(example, root).relative_to(root)}."
        )
    return records


def mask_to_boxes(mask: np.ndarray, wanted: dict[int, int], min_size: int) -> tuple[list[Box], int]:
    """Mapillary instance PNG (value = label_id * 256 + instance) -> boxes of wanted labels.

    `wanted` maps Mapillary label id -> target class id. Returns (boxes, skipped small).
    """
    h, w = mask.shape[:2]
    boxes: list[Box] = []
    small = 0
    for value in np.unique(mask):
        cls = wanted.get(int(value) // 256)
        if cls is None:
            continue
        ys, xs = np.nonzero(mask == value)
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if x2 - x1 < min_size or y2 - y1 < min_size:
            small += 1
            continue
        boxes.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return boxes, small


def scan_mapillary(m: Mapping, root: Path, target_ids: dict[str, int], stats: SourceStats) -> list[Record]:
    configs = sorted(root.rglob("config*.json"))
    if not configs:
        raise MergeError(f"{m.short_name}: no config*.json with label names in {root}")
    config = configs[-1]  # newest version, e.g. config_v2.0.json
    labels = [str(label["name"]) for label in json.loads(config.read_text())["labels"]]
    check_classes(m, labels, config.name)
    wanted = {
        i: target_ids[m.classes[name]]
        for i, name in enumerate(labels)
        if m.classes.get(name, DROP) != DROP
    }

    masks = sorted(p for p in root.rglob("*.png") if "instances" in p.parts)
    version = re.search(r"config_(v[\d.]+)\.json", config.name)
    if version and any(version.group(1) in p.parts for p in masks):
        masks = [p for p in masks if version.group(1) in p.parts]
    images = {p.stem: p for p in root.rglob("*.jpg") if "images" in p.parts}
    pairs = [(images[p.stem], p) for p in masks if p.stem in images]
    stats.missing_label_files += len(masks) - len(pairs)
    if not pairs:
        raise MergeError(f"{m.short_name}: no image + instances/*.png pairs found in {root}")

    def load(pair: tuple[Path, Path]) -> tuple[list[Box], int]:
        mask = cv2.imread(str(pair[1]), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint16:
            raise MergeError(f"{pair[1]}: expected a 16-bit single-channel instance PNG")
        return mask_to_boxes(mask, wanted, m.min_size_px)

    records = []
    for (image, mask_path), (boxes, small) in zip(pairs, threaded(load, pairs, f"scan {m.short_name}")):
        stats.small_instances_skipped += small
        records.append(Record(
            source=m.short_name,
            image=image,
            group=f"{m.short_name}/{image.stem}",
            own_test=is_own_test(image, root),
            boxes=boxes,
        ))
    return records


def scan_sources(
    sources: Sequence[Mapping], roots: Sequence[Path], target_ids: dict[str, int],
) -> tuple[list[Record], dict[str, SourceStats]]:
    records: list[Record] = []
    stats: dict[str, SourceStats] = {}
    for m in sources:
        root = find_source_dir(roots, m.folder)
        if root is None:
            raise MergeError(f"{m.short_name}: folder {m.folder} not found in {[str(r) for r in roots]}")
        st = stats[m.short_name] = SourceStats(folder=m.folder)
        scan = scan_mapillary if m.format == "mapillary" else scan_yolo
        found = scan(m, root, target_ids, st)
        st.images = len(found)
        records += found
    assign_out_stems(records)
    return records, stats


def assign_out_stems(records: Sequence[Record]) -> None:
    """`<source>_<stem>`, made unique if two files of a source share a stem."""
    taken: set[str] = set()
    for r in records:
        base = sanitize(f"{r.source}_{r.image.stem}")
        stem, n = base, 2
        while stem.lower() in taken:
            stem, n = f"{base}_{n}", n + 1
        taken.add(stem.lower())
        r.out_stem = stem


# ---------------------------------------------------------------------- build


def prepare_dir(path: Path, roots: Sequence[Path]) -> None:
    """Empty `path` for a fresh build, refusing to delete folders this script didn't make."""
    path = path.absolute()
    for raw in (r.absolute() for r in roots):
        if path == raw or raw in path.parents or path in raw.parents:
            raise MergeError(f"{path} overlaps the raw datasets folder {raw}")
    if path.exists() and any(path.iterdir()):
        if not (path / WORK_MARKER).exists():
            raise MergeError(f"Refusing to delete {path}: not made by this script. Empty it or pick another folder.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / WORK_MARKER).write_text("Made by scripts/merge_datasets.py, which may delete and rebuild this folder.\n")


def shrink_jpeg(data: bytes, ext: str, max_side: int, quality: int, where: Path) -> bytes:
    """JPEG bytes with the longer side at most `max_side`. Small JPEGs are kept as they are."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise MergeError(f"{where}: cannot decode image")
    h, w = image.shape[:2]
    if max(h, w) <= max_side and ext in (".jpg", ".jpeg"):
        return data
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise MergeError(f"{where}: JPEG encoding failed")
    return buf.tobytes()


def copy_shrink_and_hash(records: Sequence[Record], dest: Path, max_side: int, quality: int) -> tuple[int, int]:
    """Read each image once: hash it, shrink it to a JPEG of at most `max_side` px, write it.
    The md5 is of the source file. Returns (source bytes, written bytes)."""
    dest.mkdir(parents=True, exist_ok=True)

    def work(r: Record) -> tuple[int, int]:
        data = r.image.read_bytes()
        r.md5 = hashlib.md5(data).hexdigest()
        out = shrink_jpeg(data, r.image.suffix.lower(), max_side, quality, r.image)
        (dest / r.out_name).write_bytes(out)
        return len(data), len(out)

    sizes = threaded(work, records, f"copy + md5 + resize to {max_side}px")
    return sum(s for s, _ in sizes), sum(o for _, o in sizes)


def remove_duplicates(
    records: Sequence[Record], staging: Path, stats: dict[str, SourceStats], existing: Iterable[str] = (),
) -> list[Record]:
    """Keep the first copy of each exact-duplicate image (sources in mapping file order).
    `existing`: md5s already in the merged dataset."""
    seen = set(existing)
    kept = []
    for r in records:
        if r.md5 in seen:
            stats[r.source].duplicates_removed += 1
            (staging / r.out_name).unlink()
        else:
            seen.add(r.md5)
            kept.append(r)
    return kept


def assign_splits(records: Sequence[Record], ratios: Sequence[float], seed: int) -> None:
    """Grouped split per source. Groups with an image in the source's own test split go to
    test; the rest are shuffled and fill test up to its ratio, then val, then train."""
    by_source: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_source[r.source][r.group].append(r)
    for source, groups in sorted(by_source.items()):
        keys = sorted(groups)
        forced = {k for k in keys if any(r.own_test for r in groups[k])}
        rest = [k for k in keys if k not in forced]
        random.Random(f"{seed}-{source}").shuffle(rest)  # per source: adding one doesn't reshuffle others
        n_test = max(0, round(ratios[2] * len(keys)) - len(forced))
        n_val = round(ratios[1] * len(keys))
        split_of = {k: "test" for k in forced}
        for i, k in enumerate(rest):
            split_of[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for k, split in split_of.items():
            for r in groups[k]:
                r.split = split


def write_dataset(records: Sequence[Record], staging: Path, dataset: Path) -> None:
    for split in SPLITS:
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in tqdm(records, desc="write labels"):
        os.replace(staging / r.out_name, dataset / "images" / r.split / r.out_name)
        (dataset / "labels" / r.split / f"{r.out_stem}.txt").write_text(label_text(r.boxes))
    staging.rmdir()


def write_review(
    records: Sequence[Record], mappings: Sequence[Mapping], classes: Sequence[str],
    dataset: Path, review_dir: Path, samples: int, seed: int,
) -> dict[str, int]:
    """Save sample images with the review classes drawn in red (kept boxes in green)."""
    written: dict[str, int] = {}
    for m in mappings:
        for cls in m.review:
            candidates = [r for r in records if r.source == m.short_name and any(b[0] == cls for b in r.review)]
            picks = random.Random(f"{seed}-review-{m.short_name}-{cls}").sample(
                candidates, min(samples, len(candidates))
            )
            out_dir = review_dir / sanitize(cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in picks:
                image = cv2.imread(str(dataset / "images" / r.split / r.out_name))
                if image is None:
                    continue
                kept = [(classes[c], x, y, w, h) for c, x, y, w, h in r.boxes]
                review = [b for b in r.review if b[0] == cls]
                for boxes, color in ((kept, (0, 200, 0)), (review, (0, 0, 255))):
                    draw_boxes(image, boxes, color)
                cv2.imwrite(str(out_dir / r.out_name), image)
            written[f"{m.short_name}/{cls}"] = len(picks)
            logger.info("Review: %d images of %r in review/%s/", len(picks), cls, out_dir.name)
    return written


def draw_boxes(image: np.ndarray, boxes: Iterable[ReviewBox], color: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    thickness = max(2, w // 400)
    for name, x, y, bw, bh in boxes:
        p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
        p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        cv2.rectangle(image, p1, p2, color, thickness)
        cv2.putText(image, name, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def write_md5_index(path: Path, records: Iterable[Record], index: dict[str, str] | None = None) -> dict[str, str]:
    index = dict(index or {})
    index.update({r.md5: f"{r.split}/{r.out_name}" for r in records})
    path.write_text(json.dumps(index))
    return index


def load_md5_index(dataset: Path) -> dict[str, str]:
    """md5_index.json of a merged dataset, built from its images if it is missing.
    (Images that were copied unchanged have the same md5 as their source.)"""
    path = dataset / INDEX_FILE
    if path.exists():
        return json.loads(path.read_text())
    images = sorted(p for p in (dataset / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    hashes = threaded(lambda p: hashlib.md5(p.read_bytes()).hexdigest(), images, "md5 of merged images")
    index = {h: f"{p.parent.name}/{p.name}" for p, h in zip(images, hashes)}
    path.write_text(json.dumps(index))
    return index


def write_data_yaml(dataset: Path, classes: Sequence[str]) -> None:
    data = {
        "path": LOCAL_DATASET,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": list(classes),
    }
    (dataset / "data.yaml").write_text(
        "# Generated by scripts/merge_datasets.py. src/colab_data.py prepare_dataset() sets `path`.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )


# ---------------------------------------------------------------------- zip on Drive


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def free_space(target: Path) -> int:
    while not target.exists():  # the folder may not exist yet
        target = target.parent
    return shutil.disk_usage(target).free


def check_free_space(target: Path, needed: int, reclaimable: int = 0) -> None:
    """Stop if `needed` bytes (+10%) don't fit. `reclaimable`: size of a file being replaced."""
    free = free_space(target)
    if needed * SPACE_MARGIN > free + reclaimable:
        raise MergeError(
            f"Not enough space on {target}: {gb(needed)} needed (plus 10% margin), "
            f"{gb(free)} free" + (f" + {gb(reclaimable)} from the file being replaced" if reclaimable else "")
            + ". Free up Drive space (deleted files stay in Drive's Trash and still count until "
            "it is emptied), then re-run. Nothing was written."
        )
    logger.info("Space check: %s needed, %s free on %s", gb(needed), gb(free + reclaimable), target)


def make_zip(dataset: Path, dest: Path) -> int:
    """ZIP_STORED archive of `dataset` (JPEGs don't compress). Returns the file count."""
    files = sorted(p for p in dataset.rglob("*") if p.is_file() and p.name != WORK_MARKER)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in tqdm(files, desc=f"zip {dest.name}"):
            zf.write(path, path.relative_to(dataset).as_posix())
    return len(files)


def copy_with_progress(src: Path, dest: Path) -> None:
    size = src.stat().st_size
    with open(src, "rb") as fin, open(dest, "wb") as fout, tqdm(
        total=size, unit="B", unit_scale=True, desc=f"copy {src.name} -> {dest.parent.name}/"
    ) as bar:
        while chunk := fin.read(COPY_CHUNK):
            fout.write(chunk)
            bar.update(len(chunk))


def colab_sync(mountpoint: str = "/content/drive") -> None:
    """Make Drive finish its background upload: flush + unmount, then mount again.
    Outside Colab, just os.sync()."""
    os.sync()
    try:
        from google.colab import drive
    except ImportError:
        return
    logger.info("Flushing Drive (this waits until the upload has finished)...")
    drive.flush_and_unmount()
    drive.mount(mountpoint)


def verify_zip(path: Path, size: int, n_files: int) -> None:
    """The uploaded zip must have the local size, all files and valid CRCs."""
    actual = path.stat().st_size if path.exists() else 0
    if actual != size:
        raise MergeError(
            f"Upload incomplete: {path} is {gb(actual)} on Drive but {gb(size)} locally "
            "(Drive full? Check the Trash). merged.done was NOT written."
        )
    logger.info("Verifying %s on Drive (reads the whole file)...", path.name)
    with zipfile.ZipFile(path) as zf:
        if len(zf.namelist()) != n_files:
            raise MergeError(f"{path} has {len(zf.namelist())} files, expected {n_files}. merged.done was NOT written.")
        bad = zf.testzip()
    if bad is not None:
        raise MergeError(f"{path} is corrupt (first bad file: {bad}). merged.done was NOT written.")


def publish_zip(
    dataset: Path, target: ZipTarget, review_src: Path, report: dict[str, Any],
    sync: SyncFn, replace_review: bool,
) -> dict[str, Any]:
    """Zip `dataset` locally, check space, upload the zip + report + review, flush Drive,
    verify, then write merged.done. Returns the done info."""
    local_zip = dataset.parent / target.zip.name
    n_files = make_zip(dataset, local_zip)
    zip_size = local_zip.stat().st_size
    report_text = json.dumps(report, indent=2)
    review_size = dir_size(review_src) if review_src.exists() else 0
    old_zip = target.zip.stat().st_size if target.zip.exists() else 0
    logger.info("Zip: %s, %d files (%s); review images: %s", local_zip.name, n_files, gb(zip_size), gb(review_size))
    check_free_space(target.zip.parent, zip_size + len(report_text) + review_size, reclaimable=old_zip)

    target.zip.parent.mkdir(parents=True, exist_ok=True)
    target.done.unlink(missing_ok=True)  # from here until verification, the zip is not done
    copy_with_progress(local_zip, target.zip)
    target.report.write_text(report_text)
    if replace_review:
        shutil.rmtree(target.review, ignore_errors=True)
    if review_src.exists():
        copy_files(review_src, target.review, "copy review images")
    sync()
    verify_zip(target.zip, zip_size, n_files)

    info = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zip_bytes": zip_size,
        "files": n_files,
        "images": report["images"],
        "drive_free_bytes": free_space(target.zip.parent),
    }
    target.done.write_text(json.dumps(info, indent=2))
    local_zip.unlink()
    logger.info("Verified %s (%s). Drive space left: %s", target.zip, gb(zip_size), gb(info["drive_free_bytes"]))
    return info


def copy_files(src: Path, dest: Path, desc: str) -> None:
    """Copy every file under `src` into `dest`, merging folders (threaded, for Drive)."""
    files = [p for p in src.rglob("*") if p.is_file() and p.name != WORK_MARKER]
    for d in sorted({p.parent for p in files}):
        (dest / d.relative_to(src)).mkdir(parents=True, exist_ok=True)
    threaded(lambda p: shutil.copyfile(p, dest / p.relative_to(src)), files, desc)


# ---------------------------------------------------------------------- report


def source_report(
    m: Mapping, st: SourceStats, records: Iterable[Record], classes: Sequence[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "folder": st.folder,
        "mapping": dict(m.classes),  # compared on append to find changed mappings
        "unlisted": DROP if m.unlisted_drop else "fail",
        "images_found": st.images,
        "duplicates_removed": st.duplicates_removed,
        "missing_label_files": st.missing_label_files,
        "invalid_boxes": st.invalid_boxes,
        "small_instances_skipped": st.small_instances_skipped,
        "dropped_boxes": dict(st.dropped_boxes),
        "images": {s: 0 for s in SPLITS},
        "boxes": {s: {} for s in SPLITS},
    }
    for r in records:
        entry["images"][r.split] += 1
        for c, *_ in r.boxes:
            split_boxes = entry["boxes"][r.split]
            split_boxes[classes[c]] = split_boxes.get(classes[c], 0) + 1
    return entry


def finalize_report(report: dict[str, Any]) -> None:
    """Recompute the dataset totals from the per-source counts."""
    classes, sources = report["classes"], report["sources"].values()
    report["images"] = {s: sum(src["images"][s] for src in sources) for s in SPLITS}
    report["boxes"] = {
        s: {c: sum(src["boxes"][s].get(c, 0) for src in sources) for c in classes} for s in SPLITS
    }
    report["low_train_classes"] = [c for c in classes if report["boxes"]["train"][c] < MIN_TRAIN_BOXES]


def build_report(
    records: Sequence[Record], mappings: Sequence[Mapping], classes: Sequence[str],
    stats: dict[str, SourceStats], ratios: Sequence[float], seed: int, review: dict[str, int],
    sizes: dict[str, int],
) -> dict[str, Any]:
    by_source: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    report: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "classes": list(classes),
        "images": {},
        "boxes": {},
        "low_train_classes": [],
        "sizes": sizes,
        "review_samples": review,
        "sources": {
            m.short_name: source_report(m, stats[m.short_name], by_source[m.short_name], classes)
            for m in mappings
        },
    }
    finalize_report(report)
    return report


def log_summary(report: dict[str, Any]) -> None:
    """Boxes per class per split; classes under MIN_TRAIN_BOXES train boxes are flagged."""
    boxes = report["boxes"]
    logger.info("")
    logger.info("%-16s %8s %8s %8s", "class", *SPLITS)
    for c in report["classes"]:
        flag = f"   <-- under {MIN_TRAIN_BOXES} train boxes" if boxes["train"][c] < MIN_TRAIN_BOXES else ""
        logger.info("%-16s %8d %8d %8d%s", c, *(boxes[s][c] for s in SPLITS), flag)
    logger.info("%-16s %8d %8d %8d", "TOTAL boxes", *(sum(boxes[s].values()) for s in SPLITS))
    logger.info("%-16s %8d %8d %8d", "images", *(report["images"][s] for s in SPLITS))
    logger.info("")
    logger.info("%-22s %7s %6s %8s %8s", "source", "images", "dups", "dropped", "invalid")
    for name, src in report["sources"].items():
        logger.info(
            "%-22s %7d %6d %8d %8d", name, sum(src["images"].values()), src["duplicates_removed"],
            sum(src["dropped_boxes"].values()), src["invalid_boxes"],
        )
    if report["low_train_classes"]:
        logger.warning("Classes under %d train boxes: %s", MIN_TRAIN_BOXES, ", ".join(report["low_train_classes"]))


def log_sizes(source_bytes: int, merged_bytes: int) -> None:
    change = merged_bytes / source_bytes - 1 if source_bytes else 0
    logger.info("Images: %s before resize, %s after (%+.0f%%)", gb(source_bytes), gb(merged_bytes), change * 100)


# ---------------------------------------------------------------------- full merge


def merge(
    raw: RawArg,
    zip_path: Path | str,
    work: Path | str = DEFAULT_WORK,
    classes_path: Path | str = REPO / "configs" / "classes.yaml",
    mappings_dir: Path | str = REPO / "configs" / "mappings",
    ratios: Sequence[float] = (0.75, 0.15, 0.10),
    seed: int = 42,
    review_samples: int = 16,
    max_side: int = 1280,
    jpeg_quality: int = 90,
    force: bool = False,
    sync: SyncFn = colab_sync,
) -> dict[str, Any] | None:
    """Build merged.zip. Returns the report, or None if it is already done."""
    roots, target, work = as_roots(raw), ZipTarget(Path(zip_path)), Path(work)
    if target.done.exists() and not force:
        logger.info("%s is already merged and verified (%s exists). Use force to rebuild.", target.zip, target.done.name)
        return None
    check_ratios(ratios)
    for root in roots:
        if not root.is_dir():
            raise MergeError(f"Raw datasets folder not found: {root}")
    t0 = time.time()

    classes = load_classes(Path(classes_path))
    mappings = load_mappings(Path(mappings_dir), classes)
    active = active_sources(roots, mappings)
    log_mappings(mappings, active)
    for root in roots:
        logger.info("Raw datasets in %s: %s", root, gb(dir_size(root)))
    target_ids = {c: i for i, c in enumerate(classes)}

    # 1. Scan every source before copying anything, so mapping errors fail fast.
    records, stats = scan_sources(active, roots, target_ids)
    logger.info("Scanned %d images from %d sources in %.0fs", len(records), len(active), time.time() - t0)

    # 2. Copy + resize + hash, drop duplicates. 3. Split. 4. Write everything locally.
    prepare_dir(work, roots)
    dataset, review_dir, staging = work / "dataset", work / "review", work / "_staging"
    source_bytes, merged_bytes = copy_shrink_and_hash(records, staging, max_side, jpeg_quality)
    records = remove_duplicates(records, staging, stats)
    assign_splits(records, ratios, seed)
    write_dataset(records, staging, dataset)
    review = write_review(records, active, classes, dataset, review_dir, review_samples, seed)
    log_sizes(source_bytes, merged_bytes)

    sizes = {"images_source_bytes": source_bytes, "images_merged_bytes": merged_bytes, "max_side": max_side}
    report = build_report(records, active, classes, stats, ratios, seed, review, sizes)
    (dataset / REPORT_FILE).write_text(json.dumps(report, indent=2))
    write_md5_index(dataset / INDEX_FILE, records)
    write_data_yaml(dataset, classes)
    log_summary(report)

    # 5. Zip, check space, upload, flush, verify, mark done.
    publish_zip(dataset, target, review_dir, report, sync, replace_review=True)
    shutil.rmtree(work)
    logger.info("Merged dataset ready: %s (%.0fs)", target.zip, time.time() - t0)
    return report


def check_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        raise MergeError(f"ratios must be 3 numbers that sum to 1, got {list(ratios)}")


# ---------------------------------------------------------------------- append


def changed_classes(entry: dict[str, Any], m: Mapping) -> list[str]:
    """Source classes whose mapping differs from the one used when the source was merged."""
    old = entry.get("mapping")
    if old is not None:
        return sorted(c for c in set(old) | set(m.classes) if old.get(c) != m.classes.get(c))
    # Reports written before mappings were recorded: dropped classes that are now mapped.
    return sorted(c for c, n in entry.get("dropped_boxes", {}).items() if n and m.classes.get(c, DROP) != DROP)


def remap_source(
    m: Mapping, roots: Sequence[Path], dataset: Path, entry: dict[str, Any], classes: Sequence[str],
) -> tuple[dict[str, Any], int]:
    """Rewrite the merged labels of one source from its raw labels and current mapping.
    Returns (new report entry, label files changed)."""
    if m.format != "yolo":
        raise MergeError(f"{m.short_name}: re-mapping is only supported for YOLO sources")
    target_ids = {c: i for i, c in enumerate(classes)}
    records, stats = scan_sources([m], roots, target_ids)
    prefix = f"{m.short_name}_"
    merged = {
        p.stem: p for split in SPLITS
        for p in (dataset / "labels" / split).glob(f"{prefix}*.txt")
    }
    kept = [r for r in records if r.out_stem in merged]  # the others were duplicates
    for r in kept:
        r.split = merged[r.out_stem].parent.name

    def rewrite(r: Record) -> bool:
        path, text = merged[r.out_stem], label_text(r.boxes)
        if path.read_text() == text:
            return False
        path.write_text(text)
        return True

    changed = sum(threaded(rewrite, kept, f"re-map {m.short_name} labels"))
    st = stats[m.short_name]
    st.duplicates_removed = entry.get("duplicates_removed", 0)
    return source_report(m, st, kept, classes), changed


def append_to_dataset(
    roots: Sequence[Path], dataset: Path, new_sources: Sequence[str], build: Path, review_dir: Path,
    classes: Sequence[str], mappings: dict[str, Mapping], ratios: Sequence[float], seed: int,
    review_samples: int, max_side: int, jpeg_quality: int,
) -> dict[str, Any] | None:
    """Append to an unpacked merged dataset on the local disk. Returns the updated report,
    or None if there is nothing to do."""
    report = json.loads((dataset / REPORT_FILE).read_text())
    todo: list[Mapping] = []
    for name in new_sources:
        if name not in mappings:
            raise MergeError(f"{name}: no mapping file configs/mappings/{name}.yaml")
        if name in report["sources"]:
            logger.info("%s is already in the merged dataset, skipping it", name)
        else:
            todo.append(mappings[name])
    for name in report["sources"]:
        if name not in mappings:
            logger.warning("%s is merged but has no mapping file any more; its labels are kept", name)
    remaps = {
        name: changed for name, entry in report["sources"].items()
        if name in mappings and (changed := changed_classes(entry, mappings[name]))
    }
    if not todo and not remaps:
        logger.info("Nothing to append and no changed mappings.")
        return None

    shown = todo + [mappings[n] for n in remaps]
    log_mappings(shown, shown)
    for name, changed in remaps.items():
        logger.info("%s: mapping changed for %s, its merged labels will be re-mapped", name, changed)
    target_ids = {c: i for i, c in enumerate(classes)}

    records, stats = scan_sources(todo, roots, target_ids)
    index = load_md5_index(dataset)
    prepare_dir(build, roots)
    staging = build / "_staging"
    source_bytes, merged_bytes = copy_shrink_and_hash(records, staging, max_side, jpeg_quality)
    records = remove_duplicates(records, staging, stats, existing=index)
    assign_splits(records, ratios, seed)
    write_dataset(records, staging, build)
    review = write_review(records, todo, classes, build, review_dir, review_samples, seed)
    if records:
        log_sizes(source_bytes, merged_bytes)

    for name in remaps:
        report["sources"][name], n_changed = remap_source(mappings[name], roots, dataset, report["sources"][name], classes)
        logger.info("%s: %d label files re-mapped", name, n_changed)
    copy_files(build, dataset, "add new images")

    by_source: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    for m in todo:
        report["sources"][m.short_name] = source_report(m, stats[m.short_name], by_source[m.short_name], classes)
    for name, entry in report["sources"].items():
        if name in mappings:  # record what is applied now, for the next change check
            entry["mapping"] = dict(mappings[name].classes)
    report.setdefault("review_samples", {}).update(review)
    report.setdefault("appends", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": [m.short_name for m in todo],
        "remapped": remaps,
        "images": {s: sum(1 for r in records if r.split == s) for s in SPLITS},
        "images_source_bytes": source_bytes,
        "images_merged_bytes": merged_bytes,
    })
    finalize_report(report)
    write_md5_index(dataset / INDEX_FILE, records, index)
    (dataset / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report


def append(
    raw: RawArg,
    zip_path: Path | str,
    new_sources: Sequence[str],
    work: Path | str = DEFAULT_WORK,
    classes_path: Path | str = REPO / "configs" / "classes.yaml",
    mappings_dir: Path | str = REPO / "configs" / "mappings",
    ratios: Sequence[float] = (0.75, 0.15, 0.10),
    seed: int = 42,
    review_samples: int = 16,
    max_side: int = 1280,
    jpeg_quality: int = 90,
    sync: SyncFn = colab_sync,
) -> dict[str, Any] | None:
    """Add `new_sources` to the verified merged.zip and re-map merged sources whose mapping
    changed: unzip locally, append, re-zip, upload, verify. `raw` folders are searched in
    order for each source. Returns the updated report, or None if there was nothing to do."""
    roots, target, work = as_roots(raw), ZipTarget(Path(zip_path)), Path(work)
    check_ratios(ratios)
    if not (target.zip.exists() and target.done.exists()):
        raise MergeError(f"No verified merged dataset at {target.zip} ({target.done.name} missing). Run a full merge first.")
    t0 = time.time()
    classes = load_classes(Path(classes_path))
    mappings = {m.short_name: m for m in load_mappings(Path(mappings_dir), classes)}

    prepare_dir(work, roots)
    dataset, local_zip = work / "dataset", work / "merged_in.zip"
    copy_with_progress(target.zip, local_zip)
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(dataset)
    local_zip.unlink()
    merged_names = (yaml.safe_load((dataset / "data.yaml").read_text()) or {}).get("names")
    if merged_names != classes:
        raise MergeError(
            "configs/classes.yaml differs from the merged data.yaml. Class IDs would not match: "
            "run a full merge (force) instead of appending."
        )

    report = append_to_dataset(
        roots, dataset, new_sources, work / "build", work / "review", classes, mappings,
        ratios, seed, review_samples, max_side, jpeg_quality,
    )
    if report is None:
        shutil.rmtree(work)
        return None
    shutil.rmtree(work / "build")
    log_summary(report)
    publish_zip(dataset, target, work / "review", report, sync, replace_review=False)
    shutil.rmtree(work)
    logger.info("Append finished: %s (%.0fs)", target.zip, time.time() - t0)
    return report


# ---------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, type=Path, nargs="+",
                        help="raw dataset folder(s), searched in order for each source")
    parser.add_argument("--zip", required=True, type=Path, help="merged.zip to write (e.g. on Drive)")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="local build folder")
    parser.add_argument("--append", nargs="+", metavar="SOURCE",
                        help="add these sources to the existing zip instead of rebuilding it")
    parser.add_argument("--classes", type=Path, default=REPO / "configs" / "classes.yaml")
    parser.add_argument("--mappings", type=Path, default=REPO / "configs" / "mappings")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.75, 0.15, 0.10], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-samples", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=1280, help="resize images to at most this many px")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--force", action="store_true", help="rebuild even if the zip is already done")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    common = dict(work=args.work, classes_path=args.classes, mappings_dir=args.mappings, ratios=args.ratios,
                  seed=args.seed, review_samples=args.review_samples, max_side=args.max_side,
                  jpeg_quality=args.jpeg_quality)
    try:
        if args.append:
            append(args.raw, args.zip, args.append, **common)
        else:
            merge(args.raw, args.zip, force=args.force, **common)
    except MergeError as e:
        logger.error("MERGE FAILED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
