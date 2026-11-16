"""Report, sanity checks and a balanced train list for the merged dataset.

Used by notebooks/03_dataset_report.ipynb on the dataset unpacked by
src/colab_data.py prepare_dataset() (never on Drive). Pure Python + NumPy + PIL;
the notebook draws the charts.

- scan_dataset(): every box with its split, source, class, size and whether it is a
  pseudo-label (the last lines of a label file listed in pseudo_labels.csv).
- sanity_checks(): no image in two splits, class IDs in range, no empty label files in
  val/test, image counts match merge_report.json.
- balanced_train(): train image list that repeats images of rare focus classes (up to
  4x) and caps person-only images, written as train_balanced.txt + data_balanced.yaml
  next to merged.zip. Training applies them with src/colab_data.py apply_balanced().
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_IMGSZ = 640  # box sizes are measured as the model sees them at this training size
SMALL, MEDIUM = 32**2, 96**2  # COCO size bins, in pixels at TRAIN_IMGSZ
SIZES = ("small", "medium", "large")
MIN_TRAIN_BOXES = 300
FOCUS_CLASSES = ("curb", "pothole", "barrier", "crosswalk", "manhole", "pole", "bollard",
                 "scooter", "fire_hydrant", "bus")
BALANCED_TXT = "train_balanced.txt"
BALANCED_YAML = "data_balanced.yaml"
LOCAL_DATASET = "/content/datasets/merged"


@dataclass(frozen=True)
class BoxRow:
    split: str
    image: str  # file name, e.g. "rf_cone_x.jpg"
    source: str
    cls: int
    w: float  # normalized
    h: float
    pseudo: bool
    size: str  # small | medium | large at TRAIN_IMGSZ


@dataclass
class Scan:
    classes: list[str]
    images: dict[str, list[str]]  # split -> image file names
    image_classes: dict[tuple[str, str], set[int]]  # (split, image) -> classes in it
    boxes: list[BoxRow] = field(default_factory=list)
    bad_ids: list[str] = field(default_factory=list)  # "labels/train/x.txt:3: class 25"
    bad_lines: list[str] = field(default_factory=list)  # malformed lines
    empty_labels: dict[str, list[str]] = field(default_factory=dict)  # split -> images with no boxes
    pseudo_mismatch: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- scan

def source_of(stem: str, sources: Sequence[str]) -> str:
    """Merged images are named <source>_<stem>; the longest matching source wins."""
    for name in sorted(sources, key=len, reverse=True):
        if stem.startswith(name + "_"):
            return name
    return "unknown"


def size_bin(w: float, h: float, img_w: int, img_h: int, imgsz: int = TRAIN_IMGSZ) -> str:
    """COCO small/medium/large of a normalized box after resizing the image to `imgsz`."""
    scale = imgsz / max(img_w, img_h)
    area = (w * img_w * scale) * (h * img_h * scale)
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def pseudo_counts(csv_path: Path | None) -> dict[str, list[int]]:
    """"split/name" -> class IDs of the pseudo-labels added to that image, in file order."""
    counts: dict[str, list[int]] = defaultdict(list)
    if csv_path is None or not csv_path.exists():
        return counts
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            split_name = "/".join(Path(row["image"]).parts[-2:])
            counts[split_name].append(int(row["class_id"]))
    return counts


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:  # reads the header only
        return im.size


def scan_dataset(
    dataset: Path, classes: Sequence[str], sources: Sequence[str], pseudo_csv: Path | None = None,
    imgsz: int = TRAIN_IMGSZ, workers: int = 16,
) -> Scan:
    """Read every label file (and every image header, for box sizes) of a merged dataset."""
    classes = list(classes)
    pseudo = pseudo_counts(pseudo_csv)
    scan = Scan(classes=classes, images={}, image_classes={}, empty_labels={s: [] for s in SPLITS})
    for split in SPLITS:
        images = sorted(p for p in (dataset / "images" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        scan.images[split] = [p.name for p in images]
        with ThreadPoolExecutor(workers) as pool:
            sizes = list(pool.map(_image_size, images))
        for image, (img_w, img_h) in zip(images, sizes):
            label = dataset / "labels" / split / f"{image.stem}.txt"
            rel = f"labels/{split}/{label.name}"
            lines = [ln.split() for ln in label.read_text().splitlines()] if label.exists() else []
            lines = [ln for ln in lines if ln]
            added = pseudo.get(f"{split}/{image.name}", [])
            n_pseudo = len(added)
            if n_pseudo and [int(float(ln[0])) for ln in lines[-n_pseudo:]] != added:
                scan.pseudo_mismatch.append(rel)
                n_pseudo = 0
            source = source_of(image.stem, sources)
            present: set[int] = set()
            for i, parts in enumerate(lines):
                try:
                    cls = int(float(parts[0]))
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    scan.bad_lines.append(f"{rel}:{i + 1}")
                    continue
                if not 0 <= cls < len(classes):
                    scan.bad_ids.append(f"{rel}:{i + 1}: class {cls}")
                    continue
                if len(values) == 4:
                    w, h = values[2], values[3]
                elif len(values) >= 6 and len(values) % 2 == 0:
                    w, h = max(values[0::2]) - min(values[0::2]), max(values[1::2]) - min(values[1::2])
                else:
                    scan.bad_lines.append(f"{rel}:{i + 1}")
                    continue
                present.add(cls)
                scan.boxes.append(BoxRow(split, image.name, source, cls, w, h,
                                         pseudo=i >= len(lines) - n_pseudo, size=size_bin(w, h, img_w, img_h, imgsz)))
            scan.image_classes[(split, image.name)] = present
            if not present:
                scan.empty_labels[split].append(image.name)
    return scan


# ---------------------------------------------------------------------- counts

def images_per_split(scan: Scan) -> dict[str, int]:
    return {s: len(scan.images[s]) for s in SPLITS}


def boxes_per_class(scan: Scan) -> dict[str, dict[str, dict[str, int]]]:
    """split -> class -> {"original": n, "pseudo": n}."""
    out = {s: {c: {"original": 0, "pseudo": 0} for c in scan.classes} for s in SPLITS}
    for b in scan.boxes:
        out[b.split][scan.classes[b.cls]]["pseudo" if b.pseudo else "original"] += 1
    return out


def size_distribution(scan: Scan) -> dict[str, dict[str, int]]:
    """class -> {small, medium, large} over all splits."""
    out = {c: {s: 0 for s in SIZES} for c in scan.classes}
    for b in scan.boxes:
        out[scan.classes[b.cls]][b.size] += 1
    return out


def boxes_per_source(scan: Scan) -> dict[str, dict[str, int]]:
    """class -> source -> boxes (all splits, original and pseudo)."""
    out: dict[str, dict[str, int]] = {c: defaultdict(int) for c in scan.classes}
    for b in scan.boxes:
        out[scan.classes[b.cls]][b.source] += 1
    return {c: dict(v) for c, v in out.items()}


def class_table(scan: Scan, min_train_boxes: int = MIN_TRAIN_BOXES) -> list[dict[str, Any]]:
    """One row per class: train/val/test boxes, train pseudo-labels, train images, status."""
    per = boxes_per_class(scan)
    train_images = Counter(c for (split, _), cs in scan.image_classes.items() if split == "train" for c in cs)
    rows = []
    for i, c in enumerate(scan.classes):
        train = per["train"][c]["original"] + per["train"][c]["pseudo"]
        rows.append({
            "class": c,
            "train boxes": train,
            "train pseudo": per["train"][c]["pseudo"],
            "train images": train_images.get(i, 0),
            "val boxes": sum(per["val"][c].values()),
            "test boxes": sum(per["test"][c].values()),
            "status": "needs more data" if train < min_train_boxes else "ok",
        })
    return rows


def format_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    def line(r: dict[str, Any]) -> str:  # text left-aligned, numbers right-aligned
        return "  ".join(str(r[k]).rjust(widths[k]) if isinstance(r[k], (int, float)) else str(r[k]).ljust(widths[k])
                         for k in keys)

    return "\n".join([line({k: k for k in keys}), "  ".join("-" * widths[k] for k in keys), *map(line, rows)])


# ---------------------------------------------------------------------- sanity checks

@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sanity_checks(
    dataset: Path, scan: Scan, report: dict[str, Any], hash_images: bool = True, workers: int = 16,
) -> list[Check]:
    """FAIL = the dataset is wrong (fix it before training). WARN = worth a look."""
    checks: list[Check] = []

    # 1. No image in two splits: by file name, and by content (md5) if `hash_images`.
    where: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        for name in scan.images[split]:
            where[name].append(split)
    by_name = sorted(n for n, s in where.items() if len(s) > 1)
    by_content: list[str] = []
    if hash_images:
        paths = [dataset / "images" / s / n for s in SPLITS for n in scan.images[s]]
        with ThreadPoolExecutor(workers) as pool:
            hashes = list(pool.map(_md5, paths))
        splits_of: dict[str, set[str]] = defaultdict(set)
        names_of: dict[str, list[str]] = defaultdict(list)
        for p, h in zip(paths, hashes):
            splits_of[h].add(p.parent.name)
            names_of[h].append(f"{p.parent.name}/{p.name}")
        by_content = sorted(" = ".join(names_of[h]) for h, s in splits_of.items() if len(s) > 1)
    leaks = by_name + by_content
    checks.append(Check(
        "no image in two splits",
        "FAIL" if leaks else "PASS",
        f"{len(by_name)} names, {len(by_content)} identical files in more than one split"
        + (f", e.g. {leaks[0]}" if leaks else "") + ("" if hash_images else " (content not hashed)"),
    ))

    # 2. Class IDs in range, and well-formed lines.
    n = len(scan.classes)
    checks.append(Check(
        f"class IDs within 0-{n - 1}",
        "FAIL" if scan.bad_ids or scan.bad_lines else "PASS",
        f"{len(scan.bad_ids)} out-of-range IDs, {len(scan.bad_lines)} malformed lines"
        + (f", e.g. {(scan.bad_ids + scan.bad_lines)[0]}" if scan.bad_ids or scan.bad_lines else ""),
    ))

    # 3. Empty label files in val/test. They are background images (kept on purpose by the
    #    merge), not broken files, so this is a warning: they lower nothing but add no boxes.
    empty = {s: len(scan.empty_labels[s]) for s in ("val", "test")}
    checks.append(Check(
        "no empty label files in val/test",
        "WARN" if any(empty.values()) else "PASS",
        f"val {empty['val']}, test {empty['test']} images without boxes (background images)"
        + (f", e.g. {(scan.empty_labels['val'] + scan.empty_labels['test'])[0]}" if any(empty.values()) else ""),
    ))

    # 4. Image counts match merge_report.json.
    expected = {s: report.get("images", {}).get(s, 0) for s in SPLITS}
    found = images_per_split(scan)
    checks.append(Check(
        "image counts match merge_report.json",
        "PASS" if found == expected else "FAIL",
        f"found {found}, report {expected}",
    ))

    if scan.pseudo_mismatch:
        checks.append(Check(
            "pseudo_labels.csv matches the label files", "WARN",
            f"{len(scan.pseudo_mismatch)} label files don't end with their CSV boxes (counted as original), "
            f"e.g. {scan.pseudo_mismatch[0]}",
        ))
    return checks


# ---------------------------------------------------------------------- balancing

def repeat_factors(
    scan: Scan, focus: Sequence[str] = FOCUS_CLASSES, max_repeat: int = 4, target_images: int | None = None,
) -> tuple[dict[str, int], int]:
    """Repeat factor per focus class: ceil(target / train images with the class), clamped
    to 1..max_repeat. `target` defaults to the median train images per class (all classes).
    Returns (factors, target)."""
    per_class = Counter(c for (split, _), cs in scan.image_classes.items() if split == "train" for c in cs)
    counts = [per_class.get(i, 0) for i in range(len(scan.classes))]
    target = target_images if target_images is not None else int(np.median(counts)) if counts else 0
    factors = {}
    for name in focus:
        if name not in scan.classes:
            raise ValueError(f"Focus class {name!r} is not in the class list")
        n = per_class.get(scan.classes.index(name), 0)
        factors[name] = 1 if n == 0 else max(1, min(max_repeat, math.ceil(target / n)))
    return factors, target


def balanced_train(
    scan: Scan, focus: Sequence[str] = FOCUS_CLASSES, max_repeat: int = 4, person_cap: int = 4000,
    target_images: int | None = None, seed: int = 42,
) -> tuple[list[str], dict[str, Any]]:
    """Lines of train_balanced.txt ("./images/train/<name>", relative to the dataset root;
    an image repeated k times is listed k times) and a summary.

    - An image with a focus class is repeated by the largest factor of its focus classes.
    - Person-only images (every box is a person) are randomly capped at `person_cap`.
    - Every other image (background images too) is listed once.
    """
    factors, target = repeat_factors(scan, focus, max_repeat, target_images)
    focus_ids = {scan.classes.index(c): f for c, f in factors.items()}
    person = scan.classes.index("person") if "person" in scan.classes else None

    person_only = [n for n in scan.images["train"] if person is not None and scan.image_classes[("train", n)] == {person}]
    rng = random.Random(seed)
    dropped = set(rng.sample(person_only, len(person_only) - person_cap)) if len(person_only) > person_cap else set()

    repeats: dict[str, int] = {}
    for name in scan.images["train"]:
        if name in dropped:
            continue
        present = scan.image_classes[("train", name)]
        repeats[name] = max([focus_ids[c] for c in present if c in focus_ids], default=1)
    lines = [f"./images/train/{name}" for name in sorted(repeats) for _ in range(repeats[name])]

    boxes_before = Counter(scan.classes[b.cls] for b in scan.boxes if b.split == "train")
    per_image = defaultdict(Counter)
    for b in scan.boxes:
        if b.split == "train":
            per_image[b.image][scan.classes[b.cls]] += 1
    boxes_after: Counter[str] = Counter()
    for name, k in repeats.items():
        for c, n in per_image[name].items():
            boxes_after[c] += n * k
    summary = {
        "target_images": target,
        "factors": factors,
        "max_repeat": max_repeat,
        "person_cap": person_cap,
        "person_only_images": len(person_only),
        "person_only_dropped": len(dropped),
        "train_images": len(scan.images["train"]),
        "lines": len(lines),
        "repeated_images": sum(1 for k in repeats.values() if k > 1),
        "boxes_before": {c: boxes_before.get(c, 0) for c in scan.classes},
        "boxes_after": {c: boxes_after.get(c, 0) for c in scan.classes},
    }
    return lines, summary


def write_balanced(
    dest: Path, lines: Sequence[str], classes: Sequence[str], merged_done: dict[str, Any], summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Write train_balanced.txt and data_balanced.yaml into `dest` (next to merged.zip).

    data_balanced.yaml keeps merged.done's content, so apply_balanced() can refuse a list
    made for an older merged.zip.
    """
    dest.mkdir(parents=True, exist_ok=True)
    txt, yml = dest / BALANCED_TXT, dest / BALANCED_YAML
    txt.write_text("".join(f"{ln}\n" for ln in lines))
    data = {
        "path": LOCAL_DATASET,
        "train": BALANCED_TXT,
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": list(classes),
        "merged_done": merged_done,
        "balance": {k: summary[k] for k in ("target_images", "factors", "max_repeat", "person_cap",
                                            "person_only_dropped", "lines")},
    }
    yml.write_text(
        "# Written by notebooks/03_dataset_report.ipynb. Apply with src/colab_data.py apply_balanced()\n"
        "# after prepare_dataset(); it sets `path` and copies train_balanced.txt into the dataset.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    return txt, yml


def example_grid(
    dataset: Path, scan: Scan, cls: str, n: int = 16, seed: int | None = None, tile: tuple[int, int] = (400, 300),
) -> np.ndarray | None:
    """4x4 grid of random images with `cls`: its original boxes thick yellow, its
    pseudo-labels thick orange, other classes thin gray. None if no image has the class."""
    import cv2

    picks = images_with_class(scan, cls)
    if not picks:
        return None
    picks = random.Random(seed).sample(picks, min(n, len(picks)))
    idx = scan.classes.index(cls)
    by_image: dict[tuple[str, str], list[BoxRow]] = defaultdict(list)
    wanted = set(picks)
    for b in scan.boxes:
        if (b.split, b.image) in wanted:
            by_image[(b.split, b.image)].append(b)
    tw, th = tile
    tiles = []
    for split, name in picks:
        image = cv2.imread(str(dataset / "images" / split / name))
        if image is None:
            image = np.zeros((th, tw, 3), np.uint8)
        ih, iw = image.shape[:2]
        label = dataset / "labels" / split / f"{Path(name).stem}.txt"
        rows = [ln.split() for ln in label.read_text().splitlines() if ln.strip()] if label.exists() else []
        pseudo_from = len(rows) - sum(1 for b in by_image[(split, name)] if b.pseudo)
        for i, parts in enumerate(rows):
            c, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
            p1 = (int((x - w / 2) * iw), int((y - h / 2) * ih))
            p2 = (int((x + w / 2) * iw), int((y + h / 2) * ih))
            if c == idx:
                color = (0, 140, 255) if i >= pseudo_from else (0, 230, 255)
                cv2.rectangle(image, p1, p2, color, max(3, iw // 250))
            else:
                cv2.rectangle(image, p1, p2, (170, 170, 170), max(1, iw // 600))
        scale = min(tw / iw, th / ih)
        small = cv2.resize(image, (int(iw * scale), int(ih * scale)))
        canvas = np.zeros((th, tw, 3), np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        caption = f"{split} / {by_image[(split, name)][0].source}"
        cv2.putText(canvas, caption, (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


def images_with_class(scan: Scan, cls: str) -> list[tuple[str, str]]:
    """(split, image) pairs that have at least one box of `cls`."""
    idx = scan.classes.index(cls)
    return sorted(k for k, cs in scan.image_classes.items() if idx in cs)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
