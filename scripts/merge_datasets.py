"""Merge every raw dataset into one YOLO dataset with the classes in configs/classes.yaml.

Each source has a mapping file configs/mappings/<short_name>.yaml:

    folder: rf_cone              # folder under --raw
    map: {cone: traffic_cone}    # source class -> target class, or "drop"
    review: [some-class]         # optional: save sample images with these boxes drawn
    unlisted: drop               # optional: drop classes missing from `map` (default: fail)
    format: mapillary            # optional: instance masks instead of YOLO labels

Steps:
1. Scan and validate every source. Nothing is copied if a source class isn't mapped.
2. Copy the images to the local work dir while hashing them, and drop exact
   duplicates (md5).
3. Grouped train/val/test split. Roboflow augmented copies stay together, and each
   dataset's own test split stays in test.
4. Write labels, data.yaml, merge_report.json and the review images, then copy the
   result to --out and mark it with .done.

Usage (Colab, see notebooks/02_merge_datasets.ipynb):
    python scripts/merge_datasets.py \\
        --raw /content/drive/MyDrive/clear_vision/datasets/raw \\
        --out /content/drive/MyDrive/clear_vision/datasets/merged \\
        --work /content/merged_tmp
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
DONE = ".done"
WORK_MARKER = ".merge_work"  # only folders carrying this marker are ever deleted
MIN_TRAIN_BOXES = 300
MAX_MISSING_LABELS = 0.5  # fail if more than this share of a source's images lack a label file
WORKERS = 16  # threads for file I/O; Drive reads are slow but parallelize well

Box = tuple[int, float, float, float, float]  # class id, x_center, y_center, w, h (0-1)
ReviewBox = tuple[str, float, float, float, float]  # source class name + box


class MergeError(Exception):
    """A problem that would make the merged dataset wrong. The merge stops."""


@dataclass(frozen=True)
class Mapping:
    short_name: str
    folder: str
    classes: dict[str, str]  # source class -> target class or DROP
    review: tuple[str, ...] = ()
    unlisted_drop: bool = False
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
    md5: str = ""
    split: str = ""

    @property
    def out_name(self) -> str:
        return self.out_stem + self.image.suffix.lower()


@dataclass
class SourceStats:
    folder: str
    images: int = 0
    duplicates_removed: int = 0
    missing_label_files: int = 0  # kept as background images
    invalid_boxes: int = 0
    small_instances_skipped: int = 0
    dropped_boxes: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------- config


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
                format=str(data.get("format", "yolo")),
                min_size_px=int(data.get("min_size_px", 12)),
            )
        )
    return mappings


def active_sources(raw: Path, mappings: Sequence[Mapping]) -> list[Mapping]:
    """Mappings whose folder exists. Fails on missing sources and unmapped folders."""
    active = []
    for m in mappings:
        if (raw / m.folder).is_dir():
            active.append(m)
        elif m.format == "mapillary":
            logger.info("raw/%s not found: %s skipped", m.folder, m.short_name)
        else:
            raise MergeError(
                f"{m.short_name}: folder raw/{m.folder} not found. Download it, or delete "
                f"configs/mappings/{m.short_name}.yaml."
            )
    covered = {Path(m.folder).parts[0] for m in mappings}
    unmapped = sorted(
        p.name for p in raw.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in covered
    )
    if unmapped:
        raise MergeError(
            f"Folders in {raw} without a mapping file: {unmapped}. Add "
            "configs/mappings/<name>.yaml for each, so nothing is left out silently."
        )
    return active


def log_mappings(mappings: Sequence[Mapping], active: Sequence[Mapping]) -> None:
    logger.info("Class mappings (source class -> target class):")
    for m in mappings:
        present = "" if m in active else "   (not present, skipped)"
        logger.info("  %s   [raw/%s]%s", m.short_name, m.folder, present)
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
    for missing in (c for c in m.classes if c not in names):
        close = difflib.get_close_matches(missing, names, n=5, cutoff=0.4)
        raise MergeError(
            f"{m.short_name}: mapping lists class {missing!r}, but {where} has no such "
            f"class. Similar names: {close or 'none'}."
        )


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


def prepare_dir(path: Path, raw: Path) -> None:
    """Empty `path` for a fresh build, refusing to delete folders this script didn't make."""
    path = path.absolute()
    raw = raw.absolute()
    if path == raw or raw in path.parents or path in raw.parents:
        raise MergeError(f"{path} overlaps the raw datasets folder {raw}")
    if path.exists() and any(path.iterdir()):
        if not ((path / WORK_MARKER).exists() or (path / DONE).exists()):
            raise MergeError(f"Refusing to delete {path}: not made by this script. Empty it or pick another folder.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / WORK_MARKER).write_text("Made by scripts/merge_datasets.py, which may delete and rebuild this folder.\n")


def copy_and_hash(records: Sequence[Record], dest: Path) -> None:
    """Read each image once: hash it and write it to the local work dir."""
    dest.mkdir(parents=True, exist_ok=True)

    def work(r: Record) -> None:
        data = r.image.read_bytes()
        r.md5 = hashlib.md5(data).hexdigest()
        (dest / r.out_name).write_bytes(data)

    threaded(work, records, "copy + md5")


def remove_duplicates(records: Sequence[Record], staging: Path, stats: dict[str, SourceStats]) -> list[Record]:
    """Keep the first copy of each exact-duplicate image (sources in mapping file order)."""
    seen: set[str] = set()
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


def write_dataset(records: Sequence[Record], staging: Path, work: Path) -> None:
    for split in SPLITS:
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in tqdm(records, desc="write labels"):
        os.replace(staging / r.out_name, work / "images" / r.split / r.out_name)
        lines = "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in r.boxes)
        (work / "labels" / r.split / f"{r.out_stem}.txt").write_text(lines)
    staging.rmdir()


def write_review(
    records: Sequence[Record], mappings: Sequence[Mapping], classes: Sequence[str],
    work: Path, samples: int, seed: int,
) -> dict[str, int]:
    """Save sample images with the review classes drawn in red (kept boxes in green)."""
    written: dict[str, int] = {}
    for m in mappings:
        for cls in m.review:
            candidates = [r for r in records if r.source == m.short_name and any(b[0] == cls for b in r.review)]
            picks = random.Random(f"{seed}-review-{m.short_name}-{cls}").sample(
                candidates, min(samples, len(candidates))
            )
            out_dir = work / "review" / sanitize(cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in picks:
                image = cv2.imread(str(work / "images" / r.split / r.out_name))
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


# ---------------------------------------------------------------------- report


def build_report(
    records: Sequence[Record], classes: Sequence[str], stats: dict[str, SourceStats],
    ratios: Sequence[float], seed: int, review: dict[str, int],
) -> dict[str, Any]:
    images = {s: 0 for s in SPLITS}
    boxes = {s: {c: 0 for c in classes} for s in SPLITS}
    sources: dict[str, Any] = {}
    for name, st in stats.items():
        sources[name] = {
            "folder": st.folder,
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
        images[r.split] += 1
        src = sources[r.source]
        src["images"][r.split] += 1
        for c, *_ in r.boxes:
            boxes[r.split][classes[c]] += 1
            src["boxes"][r.split][classes[c]] = src["boxes"][r.split].get(classes[c], 0) + 1
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "classes": list(classes),
        "images": images,
        "boxes": boxes,
        "low_train_classes": [c for c in classes if boxes["train"][c] < MIN_TRAIN_BOXES],
        "review_samples": review,
        "sources": sources,
    }


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


def write_data_yaml(work: Path, final_root: Path, classes: Sequence[str]) -> None:
    data = {
        "path": os.path.abspath(final_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": list(classes),
    }
    (work / "data.yaml").write_text(
        "# Generated by scripts/merge_datasets.py. Do not edit; re-run the merge.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )


def publish(work: Path, out: Path, raw: Path, info: dict[str, Any]) -> None:
    """Copy the finished build to `out` (e.g. Drive), then write the .done marker."""
    if work.absolute() != out.absolute():
        prepare_dir(out, raw)
        files = [p for p in work.rglob("*") if p.is_file() and p.name != WORK_MARKER]
        for d in sorted({p.parent for p in files}):
            (out / d.relative_to(work)).mkdir(parents=True, exist_ok=True)
        threaded(lambda p: shutil.copyfile(p, out / p.relative_to(work)), files, f"copy to {out.name}")
    (out / DONE).write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------------- main


def merge(
    raw: Path | str,
    out: Path | str,
    work: Path | str | None = None,
    classes_path: Path | str = REPO / "configs" / "classes.yaml",
    mappings_dir: Path | str = REPO / "configs" / "mappings",
    ratios: Sequence[float] = (0.75, 0.15, 0.10),
    seed: int = 42,
    review_samples: int = 16,
    force: bool = False,
) -> dict[str, Any] | None:
    """Build the merged dataset. Returns the report, or None if `out` is already done."""
    raw, out = Path(raw), Path(out)
    work = Path(work) if work else out
    if (out / DONE).exists() and not force:
        logger.info("%s is already merged (%s exists). Use force to rebuild.", out, DONE)
        return None
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        raise MergeError(f"ratios must be 3 numbers that sum to 1, got {list(ratios)}")
    if not raw.is_dir():
        raise MergeError(f"Raw datasets folder not found: {raw}")
    t0 = time.time()

    classes = load_classes(Path(classes_path))
    mappings = load_mappings(Path(mappings_dir), classes)
    active = active_sources(raw, mappings)
    log_mappings(mappings, active)
    target_ids = {c: i for i, c in enumerate(classes)}

    # 1. Scan every source before copying anything, so mapping errors fail fast.
    records: list[Record] = []
    stats: dict[str, SourceStats] = {}
    for m in active:
        st = stats[m.short_name] = SourceStats(folder=m.folder)
        scan = scan_mapillary if m.format == "mapillary" else scan_yolo
        found = scan(m, raw / m.folder, target_ids, st)
        st.images = len(found)
        records += found
    assign_out_stems(records)
    logger.info("Scanned %d images from %d sources in %.0fs", len(records), len(active), time.time() - t0)

    # 2. Copy + hash, drop duplicates. 3. Split. 4. Write everything.
    prepare_dir(work, raw)
    staging = work / "_staging"
    copy_and_hash(records, staging)
    records = remove_duplicates(records, staging, stats)
    assign_splits(records, ratios, seed)
    write_dataset(records, staging, work)
    review = write_review(records, active, classes, work, review_samples, seed)

    report = build_report(records, classes, stats, ratios, seed, review)
    (work / "merge_report.json").write_text(json.dumps(report, indent=2))
    write_data_yaml(work, out, classes)
    log_summary(report)

    publish(work, out, raw, {"created": report["created"], "images": report["images"], "seed": seed})
    logger.info("Merged dataset ready: %s (%.0fs)", out / "data.yaml", time.time() - t0)
    return report


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, type=Path, help="folder with one subfolder per raw dataset")
    parser.add_argument("--out", required=True, type=Path, help="final merged dataset folder (e.g. on Drive)")
    parser.add_argument("--work", type=Path, help="local build folder, copied to --out at the end (default: --out)")
    parser.add_argument("--classes", type=Path, default=REPO / "configs" / "classes.yaml")
    parser.add_argument("--mappings", type=Path, default=REPO / "configs" / "mappings")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.75, 0.15, 0.10], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-samples", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="rebuild even if --out is already done")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        merge(args.raw, args.out, args.work, args.classes, args.mappings, args.ratios, args.seed,
              args.review_samples, args.force)
    except MergeError as e:
        logger.error("MERGE FAILED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
