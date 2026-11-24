"""Measure a trained model against the proposed safety target (notebooks/10_safety_report.ipynb).

    python scripts/safety_report.py --model models/budget_v1/best.pt --data /path/to/merged_test/data.yaml
    python scripts/safety_report.py --model /content/drive/MyDrive/clear_vision/exports/budget_v1/best.pt   # Colab

The target (configs/default.yaml `safety_target`): for each hazard class and person, at least
90% of the medium and large ground-truth boxes on the merged TEST split must be found.

- **Found** = a prediction of the same class with IoU >= 0.5. Each ground-truth box can be found
  by one prediction, highest confidence first (as in notebook 08's mAP).
- **Size** = box area as a fraction of the image: small < 1% (far away), medium 1-5%,
  large > 5% (close). Small boxes are reported but not part of PASS.
- **Precision** per class = predictions that found a box / all predictions of that class.
  Sources label only their own classes, so a correct detection of an unlabeled object counts
  as wrong: precision here is a lower bound.
- The model runs once at the lowest threshold; higher thresholds drop the weaker predictions.
  That gives the same boxes as running again: NMS only lets a stronger box remove a weaker one.
- Without --data, only the test split is extracted from merged.zip (notebook 08's helper).

Writes runs/safety_report/<model>_<date>.json (--out).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import compare_detectors as cd  # noqa: E402
import mobile_export as me  # noqa: E402

from src.detection import Detection  # noqa: E402
from src.detector import load_class_names  # noqa: E402

logger = logging.getLogger("safety_report")

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DRIVE_RUNS = Path("/content/drive/MyDrive/clear_vision/runs/safety_report")
SIZE_GROUPS = ("small", "medium", "large")
MEDIUM_LARGE = "medium+large"

GroundTruth = Mapping[str, Sequence[tuple[str, float, float, float, float]]]  # image -> (class, x1, y1, x2, y2)
Predictions = Mapping[str, Sequence[Detection]]  # image -> detections, normalized boxes


@dataclass(frozen=True)
class SafetyTarget:
    classes: tuple[str, ...]
    conf_levels: tuple[float, ...]
    iou: float = 0.5
    small_max: float = 0.01
    large_min: float = 0.05
    min_recall: float = 0.90


def load_target(config_path: Path = DEFAULT_CONFIG) -> SafetyTarget:
    """The `safety_target` section of configs/default.yaml."""
    cfg = yaml.safe_load(Path(config_path).read_text())["safety_target"]
    return SafetyTarget(
        classes=tuple(cfg["classes"]),
        conf_levels=tuple(sorted(float(c) for c in cfg["conf_levels"])),
        iou=float(cfg["iou"]),
        small_max=float(cfg["small_max"]),
        large_min=float(cfg["large_min"]),
        min_recall=float(cfg["min_recall"]),
    )


def size_group(box: Sequence[float], target: SafetyTarget) -> str:
    """small / medium / large from a normalized (x1, y1, x2, y2) box's share of the image area."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < target.small_max:
        return "small"
    return "large" if area > target.large_min else "medium"


def match(preds: Sequence[Detection], gts: Sequence[Sequence[float]], iou: float) -> tuple[list[bool], list[bool]]:
    """Match one image's predictions and ground truth of ONE class.

    Highest confidence first, each prediction takes the unused box it overlaps most; a match
    needs IoU >= `iou`. Returns (found flag per ground-truth box, hit flag per prediction).
    """
    found = [False] * len(gts)
    hit = [False] * len(preds)
    for i in sorted(range(len(preds)), key=lambda k: -preds[k].confidence):
        box = (preds[i].x1, preds[i].y1, preds[i].x2, preds[i].y2)
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if not found[j]:
                overlap = cd.box_iou(box, g)
                if overlap > best:
                    best, best_j = overlap, j
        if best_j >= 0 and best >= iou:
            found[best_j] = hit[i] = True
    return found, hit


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def evaluate(predictions: Predictions, ground_truth: GroundTruth, target: SafetyTarget, conf: float) -> dict[str, Any]:
    """Recall per size group, precision and PASS for every target class at threshold `conf`.

    Returns {"conf", "classes": {cls: {...}}, "misses": {cls: [{"image", "box", "size"}]}}.
    Misses are medium and large boxes only. Recall is None when a group has no boxes;
    "pass" is None when a class has no medium or large boxes.
    """
    classes: dict[str, Any] = {}
    misses: dict[str, list[dict[str, Any]]] = {}
    for cls in target.classes:
        n = dict.fromkeys(SIZE_GROUPS, 0)
        found_n = dict.fromkeys(SIZE_GROUPS, 0)
        n_pred = n_hit = 0
        cls_misses: list[dict[str, Any]] = []
        for image in sorted(set(ground_truth) | set(predictions)):
            gts = [g[1:] for g in ground_truth.get(image, ()) if g[0] == cls]
            preds = [d for d in predictions.get(image, ()) if d.class_name == cls and d.confidence >= conf]
            found, hit = match(preds, gts, target.iou)
            n_pred += len(preds)
            n_hit += sum(hit)
            for box, ok in zip(gts, found):
                group = size_group(box, target)
                n[group] += 1
                found_n[group] += ok
                if not ok and group != "small":
                    cls_misses.append({"image": image, "box": [round(v, 5) for v in box], "size": group})
        n_ml = n["medium"] + n["large"]
        recall = {g: _ratio(found_n[g], n[g]) for g in SIZE_GROUPS}
        recall[MEDIUM_LARGE] = _ratio(found_n["medium"] + found_n["large"], n_ml)
        classes[cls] = {
            "n": n,
            "found": found_n,
            "recall": recall,
            "precision": _ratio(n_hit, n_pred),
            "predictions": n_pred,
            "pass": None if recall[MEDIUM_LARGE] is None else recall[MEDIUM_LARGE] >= target.min_recall,
        }
        misses[cls] = cls_misses
    return {"conf": conf, "classes": classes, "misses": misses}


def failing_classes(result: Mapping[str, Any]) -> list[str]:
    return [c for c, r in result["classes"].items() if r["pass"] is False]


def format_table(result: Mapping[str, Any], target: SafetyTarget) -> str:
    """class | n small/medium/large | recall small | medium | large | medium+large | precision | result."""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    header = ["class", "n S/M/L", "rec small", "rec medium", "rec large", "rec M+L", "precision", "result"]
    rows = []
    for cls, r in result["classes"].items():
        verdict = "no data" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        rows.append([cls, "/".join(str(r["n"][g]) for g in SIZE_GROUPS),
                     *(pct(r["recall"][g]) for g in (*SIZE_GROUPS, MEDIUM_LARGE)), pct(r["precision"]), verdict])
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    lines = [f"conf {result['conf']:.2f}: PASS = medium+large recall >= {target.min_recall:.2f} (IoU {target.iou})",
             "  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(x).ljust(w) for x, w in zip(row, widths)) for row in rows]
    passed = sum(r["pass"] is True for r in result["classes"].values())
    judged = sum(r["pass"] is not None for r in result["classes"].values())
    lines.append(f"{passed} of {judged} classes pass")
    return "\n".join(lines)


# ---------------------------------------------------------------------- data and model

def load_ground_truth(pairs: Sequence[tuple[Path, Path]], names: Sequence[str], root: Path) -> dict[str, list]:
    """image id (path relative to `root`) -> [(class, x1, y1, x2, y2)] from YOLO label files."""
    gt: dict[str, list] = {}
    for image, label in pairs:
        boxes = cd.yolo_label_boxes(label)
        gt[image_id(image, root)] = [(names[c], x1, y1, x2, y2) for c, x1, y1, x2, y2 in boxes if 0 <= c < len(names)]
    return gt


def image_id(image: Path, root: Path) -> str:
    return image.relative_to(root).as_posix()


def predict(model_path: Path, images: Sequence[Path], root: Path, names: Sequence[str], conf: float,
            imgsz: int, batch: int = 16, device: str | None = None) -> dict[str, list[Detection]]:
    """Run an Ultralytics model on every image at `conf`; boxes normalized to the image."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != list(names):
        raise RuntimeError(f"{model_path} was trained on other classes than the test split:\n{model_names}\nvs\n{list(names)}")
    out: dict[str, list[Detection]] = {}
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, device=device, verbose=False)
        for path, res in zip(chunk, results):
            boxes = res.boxes
            out[image_id(path, root)] = [
                Detection(model_names[int(c)], float(s), *map(float, b))
                for b, s, c in zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
            ]
        if (start // batch) % 50 == 0:
            logger.info("predicted %d / %d images", min(start + batch, len(images)), len(images))
    return out


def model_label(model_path: Path) -> str:
    """exports/budget_v1/best.pt -> budget_v1; my_model.pt -> my_model."""
    return model_path.parent.name if model_path.stem in ("best", "last") else model_path.stem


# ---------------------------------------------------------------------- report and examples

def build_report(model_path: Path, imgsz: int, n_images: int, target: SafetyTarget,
                 results: Sequence[Mapping[str, Any]], date: str) -> dict[str, Any]:
    return {
        "model": model_label(model_path),
        "model_path": str(model_path),
        "date": date,
        "imgsz": imgsz,
        "test_images": n_images,
        "target": asdict(target),
        "thresholds": {f"{r['conf']:.2f}": r for r in results},
    }


def save_report(report: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['model']}_{report['date']}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def pick_misses(misses: Sequence[Mapping[str, Any]], n: int = 8, seed: int = 0) -> list[Mapping[str, Any]]:
    """Up to `n` misses, one per image first, chosen the same way every run."""
    first: dict[str, int] = {}
    for i, m in enumerate(misses):
        first.setdefault(m["image"], i)
    rng = random.Random(seed)
    firsts = set(first.values())
    chosen = rng.sample(sorted(firsts), min(n, len(firsts)))
    rest = [i for i in range(len(misses)) if i not in firsts]
    chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    return [misses[i] for i in chosen]


def draw_miss(image: np.ndarray, miss: Mapping[str, Any], cls: str, preds: Sequence[Detection]) -> np.ndarray:
    """RGB copy of `image`: the missed box in red, the model's predictions in yellow (with names),
    so a box found as the wrong class shows up."""
    out = image.copy()
    h, w = out.shape[:2]

    def px(box: Sequence[float]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))

    thick = max(2, round(min(h, w) / 200))
    for d in preds:
        p1, p2 = px((d.x1, d.y1, d.x2, d.y2))
        cv2.rectangle(out, p1, p2, (0, 220, 255), thick)
        cv2.putText(out, f"{d.class_name} {d.confidence:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, thick / 3, (0, 220, 255), max(1, thick // 2))
    p1, p2 = px(miss["box"])
    cv2.rectangle(out, p1, p2, (0, 0, 255), thick + 1)
    label_y = p1[1] - 6 if p1[1] > 12 * thick else min(h - 4, p2[1] + 8 * thick)  # above the box if room
    cv2.putText(out, f"MISSED {cls}", (p1[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX, thick / 3, (0, 0, 255), max(1, thick // 2))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------- command line

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="trained .pt (e.g. exports/budget_v1/best.pt)")
    ap.add_argument("--data", type=Path, help="data.yaml of an extracted test split (skips merged.zip)")
    ap.add_argument("--zip", type=Path, default=cd.DRIVE_ZIP, help="merged.zip to extract the test split from")
    ap.add_argument("--dest", type=Path, default=cd.TEST_DEST, help="where the test split is extracted")
    ap.add_argument("--imgsz", type=int, help="default: metrics.json next to the model, else 512")
    ap.add_argument("--out", type=Path, default=DRIVE_RUNS, help="folder for <model>_<date>.json")
    ap.add_argument("--max-images", type=int, help="only the first N test images (quick check)")
    ap.add_argument("--device", help="e.g. 0, cpu, mps (default: Ultralytics picks)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    target = load_target(args.config)
    names = load_class_names(ROOT / "configs" / "classes.yaml")
    data_yaml = args.data or cd.extract_test_split(args.zip, args.dest, names)
    root = Path(yaml.safe_load(data_yaml.read_text()).get("path", data_yaml.parent))
    pairs = cd.load_split(data_yaml, "test", args.max_images)
    if not pairs:
        logger.error("no test images under %s", root)
        return 2
    imgsz = args.imgsz or me.read_imgsz(args.model.parent / "metrics.json")
    logger.info("%d test images, imgsz %d, thresholds %s", len(pairs), imgsz, target.conf_levels)
    gt = load_ground_truth(pairs, names, root)
    preds = predict(args.model, [img for img, _ in pairs], root, names, min(target.conf_levels), imgsz,
                    device=args.device)
    results = [evaluate(preds, gt, target, conf) for conf in target.conf_levels]
    for r in results:
        print("\n" + format_table(r, target))
    report = build_report(args.model, imgsz, len(pairs), target, results, dt.date.today().isoformat())
    print(f"\nWrote {save_report(report, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
