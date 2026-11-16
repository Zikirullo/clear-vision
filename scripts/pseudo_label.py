"""Pseudo-label missing COCO objects in the merged dataset.

Each source labels only its own classes: people in rf_cone images are not boxed, so
training learns to ignore them and evaluation counts correct person detections as false
positives. This script runs the most accurate COCO model (yolo11x) over every image of a
non-COCO source and ADDS boxes for the COCO classes in our list:

- A box is added only if its confidence is >= --conf (0.5) and it does not overlap an
  existing box of the same class with IoU > --iou (0.5).
- Original label lines are never removed or changed; new lines are appended.
- All splits are labeled, so the test split stops punishing correct detections.

Files on Drive (next to merged.zip):

    merged.zip            re-zipped with the added boxes, uploaded and verified like append mode
    merged.done           written only after the upload was verified
    merge_report.json     + "pseudo_labels" (settings, totals) and per source "pseudo_labels_added"
    labels_original.zip   original content of every label file this step changed (backup)
    pseudo_labels.csv     one row per added box: image, source, class, confidence, box

Re-running is safe: if merge_report.json already has "pseudo_labels", nothing happens
unless --force. With --force, the original labels are first restored from
labels_original.zip, then the images are labeled again.

After appending new sources with notebook 02, re-run this with --force: appended images
are not pseudo-labeled, and a re-mapped source gets its raw labels back.

Usage (Colab, see notebooks/09_pseudo_labels.ipynb):
    python scripts/pseudo_label.py --zip /content/drive/MyDrive/clear_vision/datasets/merged.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_datasets import (  # noqa: E402
    IMAGE_EXTS,
    LOCAL_DATASET,
    PREPARED_MARKER,
    REPORT_FILE,
    SPLITS,
    Box,
    MergeError,
    ZipTarget,
    check_free_space,
    colab_sync,
    draw_boxes,
    gb,
    label_text,
    parse_label_text,
    publish_zip,
)
from src.colab_data import LOCAL_DATASET as LOCAL_UNPACK  # noqa: E402
from src.colab_data import prepare_dataset  # noqa: E402

logger = logging.getLogger("pseudo_label")

DRIVE_ZIP = Path("/content/drive/MyDrive/clear_vision/datasets/merged.zip")
BACKUP_FILE = "labels_original.zip"
CSV_FILE = "pseudo_labels.csv"
REPORT_KEY = "pseudo_labels"
DEFAULT_MODEL = ROOT / "models" / "yolo11x.pt"
COCO_SOURCES = ("coco_subset", "openimages")  # already label every COCO class we use
PSEUDO_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "bench",
                  "fire hydrant", "traffic light")
CSV_FIELDS = ["image", "split", "source", "class", "class_id", "confidence",
              "x_center", "y_center", "width", "height"]
GREEN = (0, 200, 0)
YELLOW = (0, 230, 255)

Prediction = tuple[int, float, float, float, float, float]  # our class id, confidence, x, y, w, h


@dataclass
class Change:
    """One label file that gets new boxes."""

    image: str  # "images/train/rf_cone_x.jpg"
    label: str  # "labels/train/rf_cone_x.txt"
    source: str
    split: str
    original: str  # original file content ("" if there was no label file)
    original_boxes: list[Box]
    added: list[Prediction] = field(default_factory=list)


# ---------------------------------------------------------------------- helpers

def canonical(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def class_map(model_names: dict[int, str], classes: Sequence[str], wanted: Sequence[str]) -> dict[int, int]:
    """Model class id -> our class id for the `wanted` classes ("fire hydrant" -> fire_hydrant)."""
    ours = {c: i for i, c in enumerate(classes)}
    by_name = {canonical(n): i for i, n in model_names.items()}
    mapping: dict[int, int] = {}
    for name in wanted:
        key = canonical(name)
        if key not in ours:
            raise MergeError(f"Pseudo-label class {name!r} is not in configs/classes.yaml")
        if key not in by_name:
            raise MergeError(f"Pseudo-label class {name!r} is not a class of the model")
        mapping[by_name[key]] = ours[key]
    return mapping


def source_of(stem: str, sources: Sequence[str]) -> str | None:
    """Merged images are named <source>_<stem>; the longest matching source wins
    (rf_scooter_bollard before a hypothetical rf_scooter)."""
    for name in sorted(sources, key=len, reverse=True):
        if stem.startswith(name + "_"):
            return name
    return None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two (x_center, y_center, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def boxes_to_add(
    predictions: Sequence[Prediction], existing: Sequence[Box], conf: float, iou: float
) -> tuple[list[Prediction], int]:
    """Predictions to add: confidence >= `conf` and no existing same-class box with IoU > `iou`.
    Returns (boxes to add, number skipped because of an overlap)."""
    added: list[Prediction] = []
    overlaps = 0
    for p in predictions:
        if p[1] < conf:
            continue
        if any(b[0] == p[0] and box_iou(b[1:], p[2:]) > iou for b in existing):
            overlaps += 1
            continue
        added.append(p)
    return added, overlaps


def predict(
    model: Any, images: Sequence[Path], id_map: dict[int, int], conf: float, imgsz: int, batch: int,
    device: str | None,
) -> Iterator[tuple[Path, list[Prediction]]]:
    """Yield (image, predictions in our class ids) for every image, `batch` images at a time."""
    kwargs: dict[str, Any] = {"conf": conf, "imgsz": imgsz, "classes": sorted(id_map), "verbose": False}
    if device:
        kwargs["device"] = device
        kwargs["half"] = device.startswith("cuda") or device.isdigit()
    for start in range(0, len(images), batch):
        chunk = list(images[start:start + batch])
        results = model.predict([str(p) for p in chunk], **kwargs)
        for path, result in zip(chunk, results):
            boxes = result.boxes
            preds: list[Prediction] = []
            if boxes is not None and len(boxes):
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                preds = [
                    (id_map[int(c)], float(p), *(float(v) for v in b))
                    for b, p, c in zip(xywhn, confs, cls_ids)
                    if int(c) in id_map
                ]
            yield path, preds


# ---------------------------------------------------------------------- backup and restore

def read_backup(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def restore_originals(dataset: Path, backup: dict[str, bytes]) -> int:
    """Write the original label files back. Returns how many were restored."""
    for name, data in backup.items():
        if not name.startswith("labels/") or ".." in Path(name).parts:
            raise MergeError(f"Unexpected file {name!r} in {BACKUP_FILE}")
        (dataset / name).write_bytes(data)
    return len(backup)


def strip_pseudo_report(report: dict[str, Any]) -> None:
    report.pop(REPORT_KEY, None)
    for entry in report.get("sources", {}).values():
        entry.pop("pseudo_labels_added", None)


def write_backup(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            zf.writestr(name, files[name])


def write_csv(path: Path, changes: Sequence[Change], classes: Sequence[str]) -> int:
    rows = 0
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ch in changes:
            for cls, conf, x, y, w, h in ch.added:
                writer.writerow({
                    "image": ch.image, "split": ch.split, "source": ch.source, "class": classes[cls],
                    "class_id": cls, "confidence": f"{conf:.4f}", "x_center": f"{x:.6f}",
                    "y_center": f"{y:.6f}", "width": f"{w:.6f}", "height": f"{h:.6f}",
                })
                rows += 1
    return rows


# ---------------------------------------------------------------------- report and display

def added_counts(changes: Sequence[Change], classes: Sequence[str]) -> dict[str, dict[str, int]]:
    """source -> class -> boxes added."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ch in changes:
        for p in ch.added:
            counts[ch.source][classes[p[0]]] += 1
    return {s: dict(c) for s, c in counts.items()}


def format_added_table(added: dict[str, dict[str, int]], classes: Sequence[str]) -> str:
    """Boxes added per source (rows) and class (columns), with totals."""
    cols = [c for c in classes if any(c in per for per in added.values())]
    header = ["source", *cols, "total"]
    rows = [[s, *(str(added[s].get(c, 0)) for c in cols), str(sum(added[s].values()))] for s in sorted(added)]
    rows.append(["TOTAL", *(str(sum(per.get(c, 0) for per in added.values())) for c in cols),
                 str(sum(sum(per.values()) for per in added.values()))])
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    fmt = lambda r: "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([fmt(header), "  ".join("-" * w for w in widths), *map(fmt, rows)])


def changes_from_csv(dataset: Path, csv_path: Path, classes: Sequence[str]) -> list[Change]:
    """Rebuild the changes of an earlier run from pseudo_labels.csv and the current labels,
    e.g. to show the grid again. The added boxes are the label file's last lines."""
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            by_image[row["image"]].append(row)
    changes = []
    for image, rows in by_image.items():
        label = Path(image.replace("images/", "labels/", 1)).with_suffix(".txt")
        text = (dataset / label).read_text() if (dataset / label).exists() else ""
        boxes, _ = parse_label_text(text, len(classes), dataset / label)
        added = [(int(r["class_id"]), float(r["confidence"]), float(r["x_center"]), float(r["y_center"]),
                  float(r["width"]), float(r["height"])) for r in rows]
        changes.append(Change(image=image, label=label.as_posix(), source=rows[0]["source"], split=rows[0]["split"],
                              original="", original_boxes=boxes[: max(0, len(boxes) - len(added))], added=added))
    return changes


def make_grid(
    dataset: Path, changes: Sequence[Change], classes: Sequence[str], n: int = 16, seed: int = 0,
    tile: tuple[int, int] = (480, 360),
) -> np.ndarray | None:
    """4 columns of random changed images: original boxes green, added boxes yellow."""
    picks = random.Random(seed).sample(list(changes), min(n, len(changes)))
    if not picks:
        return None
    tw, th = tile
    tiles = []
    for ch in picks:
        image = cv2.imread(str(dataset / ch.image))
        if image is None:
            image = np.zeros((th, tw, 3), np.uint8)
        draw_boxes(image, [(classes[c], x, y, w, h) for c, x, y, w, h in ch.original_boxes], GREEN)
        draw_boxes(image, [(f"{classes[c]} {p:.2f}", x, y, w, h) for c, p, x, y, w, h in ch.added], YELLOW)
        scale = min(tw / image.shape[1], th / image.shape[0])
        small = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        canvas = np.zeros((th, tw, 3), np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        cv2.putText(canvas, ch.source, (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


# ---------------------------------------------------------------------- main step

def pseudo_label(
    zip_path: Path | str = DRIVE_ZIP,
    dataset: Path | str = LOCAL_UNPACK,
    model: Any | None = None,
    model_path: Path | str = DEFAULT_MODEL,
    conf: float = 0.5,
    iou: float = 0.5,
    imgsz: int = 640,
    batch: int = 32,
    device: str | None = None,
    coco_sources: Sequence[str] = COCO_SOURCES,
    pseudo_classes: Sequence[str] = PSEUDO_CLASSES,
    force: bool = False,
    sync: Callable[[], None] = colab_sync,
    check_dataset: bool = True,
    grid_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Add pseudo-labels to the verified merged.zip and publish it. Returns the report's
    "pseudo_labels" entry, or None if it was already done (and not `force`)."""
    t0 = time.time()
    target, dataset = ZipTarget(Path(zip_path)), Path(dataset)
    backup_drive, csv_drive = target.zip.with_name(BACKUP_FILE), target.zip.with_name(CSV_FILE)
    if not (target.zip.exists() and target.done.exists()):
        raise MergeError(f"No verified merged dataset at {target.zip} ({target.done.name} missing). Run notebook 02.")

    report = json.loads(target.report.read_text())
    if report.get(REPORT_KEY) and not force:
        logger.info("%s already has pseudo-labels (%s). Set force to redo them.", target.report.name,
                    report[REPORT_KEY].get("created", "?"))
        return None

    data_yaml = prepare_dataset(target.zip, dataset, check=check_dataset)
    # From here on the local labels differ from merged.zip: without the marker, a crash makes
    # the next prepare_dataset() unpack the zip again instead of reusing half-edited labels.
    (dataset / PREPARED_MARKER).unlink(missing_ok=True)
    data = yaml.safe_load(data_yaml.read_text())
    classes = [str(n) for n in (data["names"].values() if isinstance(data["names"], dict) else data["names"])]

    old_backup: dict[str, bytes] = {}
    if report.get(REPORT_KEY):
        if not backup_drive.exists():
            raise MergeError(f"{target.report.name} has pseudo-labels but {backup_drive} is missing: "
                             "the originals cannot be restored. Rebuild merged.zip with notebook 02.")
        old_backup = read_backup(backup_drive)
        logger.info("Force: restored %d original label files from %s", restore_originals(dataset, old_backup),
                    backup_drive.name)
        strip_pseudo_report(report)

    if model is None:
        from ultralytics import YOLO

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_path))
    id_map = class_map(dict(model.names), classes, pseudo_classes)
    if device is None:
        import torch

        device = "0" if torch.cuda.is_available() else None

    sources = list(report["sources"])
    images: list[Path] = []
    unknown = 0
    for split in SPLITS:
        for p in sorted((dataset / "images" / split).glob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            src = source_of(p.stem, sources)
            if src is None:
                unknown += 1
            elif src not in coco_sources:
                images.append(p)
    if unknown:
        logger.warning("%d images match no source in the report; they are left alone", unknown)
    logger.info("Pseudo-labeling %d images from %d non-COCO sources (skipping %s) with %s, conf >= %.2f, IoU > %.2f",
                len(images), len({source_of(p.stem, sources) for p in images}), ", ".join(coco_sources),
                Path(str(getattr(model, "ckpt_path", model_path))).name, conf, iou)

    changes: list[Change] = []
    overlaps = 0
    for path, preds in tqdm(predict(model, images, id_map, conf, imgsz, batch, device), total=len(images),
                            desc="pseudo-label"):
        split = path.parent.name
        label = dataset / "labels" / split / f"{path.stem}.txt"
        original = label.read_text() if label.exists() else ""
        existing, _ = parse_label_text(original, len(classes), label)
        added, skipped = boxes_to_add(preds, existing, conf, iou)
        overlaps += skipped
        if added:
            changes.append(Change(
                image=f"images/{split}/{path.name}", label=f"labels/{split}/{label.name}",
                source=source_of(path.stem, sources) or "", split=split, original=original,
                original_boxes=existing, added=added,
            ))

    # Append only: the original text stays byte for byte at the top of the file.
    for ch in changes:
        prefix = ch.original if not ch.original or ch.original.endswith("\n") else ch.original + "\n"
        (dataset / ch.label).write_text(prefix + label_text([(c, x, y, w, h) for c, _, x, y, w, h in ch.added]))

    added = added_counts(changes, classes)
    by_split = {s: {} for s in SPLITS}
    for ch in changes:
        for p in ch.added:
            by_split[ch.split][classes[p[0]]] = by_split[ch.split].get(classes[p[0]], 0) + 1
    entry = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": Path(str(getattr(model, "ckpt_path", model_path))).name,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "classes": [canonical(c) for c in pseudo_classes],
        "skipped_sources": list(coco_sources),
        "images_checked": len(images),
        "label_files_changed": len(changes),
        "boxes_added": sum(len(ch.added) for ch in changes),
        "skipped_overlap": overlaps,
        "added": added,
        "added_by_split": by_split,
    }
    report[REPORT_KEY] = entry
    for name, src in report["sources"].items():
        if name in added:
            src["pseudo_labels_added"] = added[name]
    (dataset / REPORT_FILE).write_text(json.dumps(report, indent=2))
    data["path"] = LOCAL_DATASET  # as merge_datasets writes it; prepare_dataset() sets the real path
    (dataset / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    logger.info("\nBoxes added per source and class (%d label files changed, %d predictions skipped as "
                "overlapping an existing box):\n%s\n", len(changes), overlaps, format_added_table(added, classes))

    # Backup = originals of every file that differs from its original: this run's changes,
    # plus files changed by an earlier run (already restored), so a later --force can always undo.
    work = dataset.parent / "pseudo_tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    backup = dict(old_backup)
    for ch in changes:
        backup.setdefault(ch.label, ch.original.encode())
    backup_local, csv_local = work / BACKUP_FILE, work / CSV_FILE
    write_backup(backup_local, backup)
    rows = write_csv(csv_local, changes, classes)

    if grid_path is not None:
        grid = make_grid(dataset, changes, classes)
        if grid is not None:
            cv2.imwrite(str(grid_path), grid)

    # Backup and CSV first, flushed and verified, so the zip on Drive never has pseudo-labels
    # without a backup that can undo them.
    extras = backup_local.stat().st_size + csv_local.stat().st_size
    check_free_space(target.zip.parent, extras)
    shutil.copyfile(backup_local, backup_drive)
    shutil.copyfile(csv_local, csv_drive)
    sync()
    if backup_drive.stat().st_size != backup_local.stat().st_size or read_backup(backup_drive) != backup:
        raise MergeError(f"{backup_drive} did not upload correctly. merged.zip was NOT changed.")
    logger.info("Uploaded %s (%d files) and %s (%d rows)", BACKUP_FILE, len(backup), CSV_FILE, rows)

    info = publish_zip(dataset, target, work / "no_review", report, sync, replace_review=False)
    (dataset / PREPARED_MARKER).write_text(json.dumps(info))  # prepare_dataset() reuses this unpack
    shutil.rmtree(work)
    logger.info("Pseudo-labels finished: %d boxes added, %s verified (%s) in %.0fs",
                entry["boxes_added"], target.zip.name, gb(info["zip_bytes"]), time.time() - t0)
    return entry


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=DRIVE_ZIP, help="verified merged.zip (with merged.done)")
    parser.add_argument("--dataset", type=Path, default=LOCAL_UNPACK, help="local unpack folder")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="COCO model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="minimum confidence of an added box")
    parser.add_argument("--iou", type=float, default=0.5, help="skip a box overlapping a same-class label above this")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    parser.add_argument("--coco-sources", nargs="+", default=list(COCO_SOURCES), help="sources not to label")
    parser.add_argument("--force", action="store_true", help="restore the originals and label again")
    parser.add_argument("--grid", type=Path, help="save a grid of 16 changed images here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        pseudo_label(args.zip, args.dataset, model_path=args.model, conf=args.conf, iou=args.iou,
                     imgsz=args.imgsz, batch=args.batch, device=args.device, coco_sources=args.coco_sources,
                     force=args.force, grid_path=args.grid)
    except MergeError as e:
        logger.error("PSEUDO-LABELING FAILED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
