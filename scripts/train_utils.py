"""Helpers for notebooks/05_train_yolo.ipynb: presets, train subset, epoch timer,
test-split comparison with notebook 08, per-class choice, missed objects.

Kept out of the notebook so they are unit-tested (tests/test_train_utils.py).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

PRESETS: dict[str, dict[str, Any]] = {
    # Light fine-tune of a COCO-pretrained model on the free Colab T4 (~1-2 h, resumable).
    "BUDGET": {"MODEL": "yolo11n.pt", "IMGSZ": 512, "EPOCHS": 30, "SUBSET_FRACTION": 0.3, "FREEZE": 10,
               "BATCH": -1, "PATIENCE": 10},
    "NORMAL": {"MODEL": "yolo11n.pt", "IMGSZ": 640, "EPOCHS": 50, "SUBSET_FRACTION": 0.5, "FREEZE": None,
               "BATCH": -1, "PATIENCE": 15},
}
FOCUS_CLASSES = ("curb", "stairs", "pothole", "manhole", "bollard", "barrier", "traffic_cone", "pole",
                 "scooter", "crosswalk")
CRITICAL_CLASSES = ("curb", "stairs", "pothole", "manhole", "pole")  # recall matters most (falls, collisions)
# Classes the hybrid backend takes from the COCO model; the rest come from the open-vocab model.
COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "traffic_light", "bench",
                "fire_hydrant")

# Augmentations for chest/glasses-mounted walking footage: colour and light change a lot,
# the camera tilts a little, objects appear at many distances; the world is never upside down.
AUGMENT: dict[str, Any] = {
    "hsv_h": 0.015, "hsv_s": 0.6, "hsv_v": 0.4,
    "degrees": 5.0, "translate": 0.1, "scale": 0.5,
    "fliplr": 0.5, "flipud": 0.0,
    "mosaic": 1.0, "close_mosaic": 5, "mixup": 0.0,
}


# ---------------------------------------------------------------------- data

def train_lines(dataset: Path, balanced: bool) -> list[str]:
    """Train list as "./images/train/<name>" lines: train_balanced.txt (repeats included)
    if `balanced`, else every train image once."""
    if balanced:
        return [ln for ln in (dataset / "train_balanced.txt").read_text().splitlines() if ln]
    images = sorted(p.name for p in (dataset / "images" / "train").iterdir() if p.is_file())
    return [f"./images/train/{name}" for name in images]


def train_image_classes(dataset: Path, n_classes: int) -> dict[str, set[int]]:
    """Image file name -> class IDs in its train label file (empty set for background)."""
    out: dict[str, set[int]] = {}
    for image in (dataset / "images" / "train").iterdir():
        label = dataset / "labels" / "train" / f"{image.stem}.txt"
        ids: set[int] = set()
        if label.exists():
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts and 0 <= (c := int(float(parts[0]))) < n_classes:
                    ids.add(c)
        out[image.name] = ids
    return out


def build_subset(
    lines: Sequence[str], image_classes: Mapping[str, set[int]], classes: Sequence[str], fraction: float,
    focus: Sequence[str] = FOCUS_CLASSES, person_cap: int = 2000, seed: int = 42,
) -> tuple[list[str], dict[str, Any]]:
    """Subset of a train list (lines may repeat an image, as in train_balanced.txt).

    - Every image with a focus class is kept, with all its repeats.
    - Of the other images, `fraction` is kept (random, seeded).
    - Person-only images among the kept ones are capped at `person_cap`.
    Val and test are never touched.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    focus_ids = {classes.index(c) for c in focus if c in classes}
    person = classes.index("person") if "person" in classes else None
    names = sorted({ln.rsplit("/", 1)[1] for ln in lines})
    rng = random.Random(seed)

    focus_imgs = [n for n in names if image_classes.get(n, set()) & focus_ids]
    others = [n for n in names if not image_classes.get(n, set()) & focus_ids]
    kept_others = set(rng.sample(others, round(len(others) * fraction)))
    person_only = sorted(n for n in kept_others if person is not None and image_classes.get(n) == {person})
    dropped = set(rng.sample(person_only, len(person_only) - person_cap)) if len(person_only) > person_cap else set()
    keep = set(focus_imgs) | (kept_others - dropped)

    out = [ln for ln in lines if ln.rsplit("/", 1)[1] in keep]
    summary = {
        "source_lines": len(lines),
        "source_images": len(names),
        "focus_images": len(focus_imgs),
        "other_images": len(others),
        "other_kept": len(kept_others) - len(dropped),
        "person_only_dropped": len(dropped),
        "subset_images": len(keep),
        "subset_lines": len(out),
        "fraction": fraction,
    }
    return out, summary


def write_subset(dataset: Path, base_yaml: Path, lines: Sequence[str]) -> Path:
    """train_subset.txt + data_subset.yaml (train = the subset, val/test unchanged) in `dataset`."""
    (dataset / "train_subset.txt").write_text("".join(f"{ln}\n" for ln in lines))
    data = yaml.safe_load(base_yaml.read_text())
    data["path"] = str(dataset.absolute())
    data["train"] = "train_subset.txt"
    out = dataset / "data_subset.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


# ---------------------------------------------------------------------- training

class EpochTimer:
    """Ultralytics on_fit_epoch_end callback: after the first epoch of this session, print
    how long it took and an estimate for the remaining epochs."""

    def __init__(self) -> None:
        self.start = time.time()
        self.first: float | None = None
        self.messages: list[str] = []

    def __call__(self, trainer: Any) -> None:
        if self.first is not None:
            return
        self.first = time.time() - self.start
        done = trainer.epoch + 1
        remaining = max(0, trainer.epochs - done)
        msg = (f"Epoch {done}/{trainer.epochs} took {self.first / 60:.1f} min (incl. validation). "
               f"Estimate for the remaining {remaining} epochs: {self.first * remaining / 3600:.1f} h "
               f"(early stopping may end sooner).")
        self.messages.append(msg)
        print(f"\n⏱  {msg}\n")


def last_checkpoint(runs_dir: Path, run_name: str) -> Path | None:
    last = runs_dir / run_name / "weights" / "last.pt"
    return last if last.exists() else None


# ---------------------------------------------------------------------- evaluation

def predict_split(detector: Any, pairs: Sequence[tuple[Path, Path]], classes: Sequence[str]) -> tuple[dict, dict]:
    """Run `detector` on (image, label) pairs. Returns (predictions, ground truth) keyed by
    image path, in the format of compare_detectors.ap50_per_class()."""
    import cv2

    from scripts.compare_detectors import yolo_label_boxes

    predictions: dict[str, list] = {}
    ground_truth: dict[str, list] = {}
    for img_path, label_path in pairs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        key = str(img_path)
        predictions[key] = detector.detect(frame)
        ground_truth[key] = [(classes[c], *box) for c, *box in yolo_label_boxes(label_path) if c < len(classes)]
    return predictions, ground_truth


@dataclass(frozen=True)
class Miss:
    image: str
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    best_iou: float  # with any prediction of the same class (0 = nothing near)
    confused_as: str | None  # class of the best-overlapping other-class prediction, if IoU >= 0.5


def find_misses(
    predictions: Mapping[str, Sequence[Any]], ground_truth: Mapping[str, Sequence[tuple]],
    classes_of_interest: Sequence[str], conf: float = 0.25, iou: float = 0.5,
) -> list[Miss]:
    """Ground-truth boxes of `classes_of_interest` with no same-class prediction at
    confidence >= `conf` and IoU >= `iou` (false negatives at a deployment-like threshold)."""
    from scripts.compare_detectors import box_iou

    misses = []
    for image, gts in ground_truth.items():
        preds = [d for d in predictions.get(image, []) if d.confidence >= conf]
        for cls, *box in gts:
            if cls not in classes_of_interest:
                continue
            same = [box_iou(box, (d.x1, d.y1, d.x2, d.y2)) for d in preds if d.class_name == cls]
            best = max(same, default=0.0)
            if best >= iou:
                continue
            other = max(((box_iou(box, (d.x1, d.y1, d.x2, d.y2)), d.class_name) for d in preds if d.class_name != cls),
                        default=(0.0, None))
            misses.append(Miss(image, cls, tuple(box), best, other[1] if other[0] >= iou else None))
    return misses


def miss_grid(misses: Sequence[Miss], predictions: Mapping[str, Sequence[Any]], n: int = 12, seed: int = 0,
              tile: tuple[int, int] = (400, 300), conf: float = 0.25) -> np.ndarray | None:
    """4 columns of random misses: the missed box red, predictions thin gray."""
    import cv2

    picks = random.Random(seed).sample(list(misses), min(n, len(misses)))
    if not picks:
        return None
    tw, th = tile
    tiles = []
    for m in picks:
        image = cv2.imread(m.image)
        if image is None:
            image = np.zeros((th, tw, 3), np.uint8)
        h, w = image.shape[:2]
        for d in predictions.get(m.image, []):
            if d.confidence >= conf:
                cv2.rectangle(image, (int(d.x1 * w), int(d.y1 * h)), (int(d.x2 * w), int(d.y2 * h)), (170, 170, 170), 1)
        x1, y1, x2, y2 = m.box
        cv2.rectangle(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (40, 40, 230), max(3, w // 250))
        scale = min(tw / w, th / h)
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
        canvas = np.zeros((th, tw, 3), np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        caption = f"missed {m.cls}" + (f" (seen as {m.confused_as})" if m.confused_as else f" (best IoU {m.best_iou:.2f})")
        cv2.putText(canvas, caption, (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


# ---------------------------------------------------------------------- comparison and choice

def load_zero_training(path: Path, backend: str = "hybrid") -> dict[str, float | None]:
    """Per-class mAP50 of a backend from notebook 08's results JSON (None = not measured)."""
    data = json.loads(path.read_text())
    return dict(data["map50"][backend])


def _num(v: float | None) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def comparison_rows(
    classes: Sequence[str], zero: Mapping[str, float | None], finetuned: Mapping[str, float | None],
    recall: Mapping[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    """class | zero-training hybrid | fine-tuned | better? | recall (fine-tuned)."""
    rows = []
    for c in classes:
        z, f = _num(zero.get(c)), _num(finetuned.get(c))
        better = "n/a" if f is None or z is None else "yes" if f > z else "same" if f == z else "no"
        rows.append({"class": c, "zero-training hybrid": z, "fine-tuned": f, "better?": better,
                     "recall": _num((recall or {}).get(c)), "critical": c in CRITICAL_CLASSES})
    return rows


def format_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(v: Any) -> str:
        return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    header = ["class", "zero-training hybrid", "fine-tuned", "better?", "recall"]
    body = [[("! " if r["critical"] else "  ") + r["class"], *(cell(r[k]) for k in header[1:])] for r in rows]
    widths = [max(len(x[i]) for x in [header, *body]) for i in range(len(header))]
    fmt = lambda r: "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([fmt(header), "  ".join("-" * w for w in widths), *map(fmt, body)]) + \
        "\n! critical class: recall (share of real objects found) matters most."


def per_class_choice(
    classes: Sequence[str], zero: Mapping[str, float | None], finetuned: Mapping[str, float | None],
    coco_classes: Sequence[str] = (),
) -> dict[str, Any]:
    """For each class: "finetuned" or "openvocab" (the zero-training path), whichever scored
    higher on the test split. A class only one model has a score for goes to that model.
    Ties and classes neither has a score for keep the fine-tuned model, which was trained
    on these exact classes."""
    choice, scores = {}, {}
    for c in classes:
        z, f = _num(zero.get(c)), _num(finetuned.get(c))
        use = "openvocab" if z is not None and (f is None or z > f) else "finetuned"
        choice[c] = use
        scores[c] = {"finetuned": f, "zero_training": z,
                     "zero_training_model": "coco" if c in coco_classes else "openvocab"}
    return {"choice": choice, "scores": scores}


def check_space(path: Path, needed_bytes: int) -> int:
    """Free bytes at `path`; raises if `needed_bytes` (+10%) don't fit."""
    import shutil

    probe = path
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed_bytes * 1.1 > free:
        raise RuntimeError(f"Only {free / 1e9:.2f} GB free on {probe}, need {needed_bytes * 1.1 / 1e9:.2f} GB. "
                           "Free Drive space (empty the Trash too) and re-run.")
    return free


def per_class_from_ultralytics(metrics: Any, classes: Sequence[str]) -> dict[str, dict[str, float]]:
    """class -> {precision, recall, map50, map50_95} from an Ultralytics DetMetrics."""
    out: dict[str, dict[str, float]] = {}
    names = metrics.names
    for i, idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        out[str(names[int(idx)])] = {"precision": float(p), "recall": float(r), "map50": float(ap50),
                                     "map50_95": float(ap)}
    return {c: out[c] for c in classes if c in out}


def group_misses(misses: Sequence[Miss]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for m in misses:
        counts[m.cls] += 1
    return dict(counts)
