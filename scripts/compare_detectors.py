"""Compare the COCO-only, open-vocab and hybrid detectors.

Speed and counts on your own images/videos (any machine):
    python scripts/compare_detectors.py photos/ walk.mp4
    python scripts/compare_detectors.py photos/ --backends openvocab hybrid --openvocab-every 1

Prints, per backend: frames, FPS and how many detections of each class.
Hybrid reuses open-vocab results between runs (every Nth frame, as live), which
is right for a video. For a folder of unrelated photos pass --openvocab-every 1.

mAP50 per class on the merged TEST split (Colab):
    python scripts/compare_detectors.py --eval
If MyDrive/clear_vision/datasets/merged.zip and merged.done exist (Colab), only
images/test, labels/test and data.yaml are extracted from the zip into
/content/datasets/merged_test/ (not the whole 5.7 GB), and each backend is
scored with thresholds lowered to `--eval-conf`. The open-vocab model runs on
every image. The AP is COCO-style 101-point interpolated AP at IoU 0.5, computed
here because hybrid isn't one Ultralytics model that model.val() could score.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection import Detection  # noqa: E402
from src.detector import DetectorBackend, HybridDetector, build_detector, load_class_names  # noqa: E402

logger = logging.getLogger("compare_detectors")

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DRIVE_ZIP = Path("/content/drive/MyDrive/clear_vision/datasets/merged.zip")
TEST_DEST = Path("/content/datasets/merged_test")
EXTRACTED = ".extracted"  # holds merged.done's content; matching means already extracted

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Names in the printed tables (and their order).
LABELS = {"ultralytics": "COCO-only", "onnx": "COCO-only (onnx)", "openvocab": "openvocab", "hybrid": "hybrid"}
RARE_CLASSES = ("curb", "stairs", "pothole", "manhole", "bollard", "barrier",
                "traffic_cone", "pole", "scooter", "crosswalk")


# ---------------------------------------------------------------------- frames

def iter_frames(sources: Iterable[str | Path], max_frames: int | None = None) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from image files, folders of images and videos.

    `max_frames` caps the frames taken from each video.
    """
    for source in sources:
        path = Path(source)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                frame = cv2.imread(str(f))
                if frame is None:
                    logger.warning("Cannot read image %s", f)
                    continue
                yield f.name, frame
            elif ext in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(f))
                n = 0
                while max_frames is None or n < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield f"{f.name}#{n}", frame
                    n += 1
                cap.release()
            elif not path.is_dir():
                logger.warning("Skipping %s: not an image or video", f)


def benchmark(detector: DetectorBackend, frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Time detect() over `frames`; return fps, mean/max ms and per-class counts."""
    counts: Counter[str] = Counter()
    times: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        detections = detector.detect(frame)
        times.append(time.perf_counter() - start)
        counts.update(d.class_name for d in detections)
    total = sum(times)
    return {
        "frames": len(frames),
        "fps": len(frames) / total if total else 0.0,
        "mean_ms": 1000 * total / len(frames) if frames else 0.0,
        "max_ms": 1000 * max(times) if times else 0.0,
        "counts": dict(counts),
    }


# ---------------------------------------------------------------------- mAP50

def yolo_label_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file as (class_id, x1, y1, x2, y2), normalized.

    Lines are `cls cx cy w h`, or `cls x1 y1 x2 y2 x3 y3 ...` (polygon; its bounding box is used).
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, values = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        else:
            xs, ys = values[0::2], values[1::2]
            boxes.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(tp: Sequence[bool], confs: Sequence[float], n_gt: int) -> float:
    """COCO-style 101-point interpolated AP from per-prediction TP flags."""
    if n_gt == 0:
        return math.nan
    if not tp:
        return 0.0
    order = np.argsort(-np.asarray(confs), kind="stable")
    hits = np.asarray(tp, dtype=float)[order]
    tp_cum = np.cumsum(hits)
    fp_cum = np.cumsum(1.0 - hits)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    # Precision envelope: best precision at this recall or higher.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def ap50_per_class(
    predictions: Mapping[str, Sequence[Detection]],
    ground_truth: Mapping[str, Sequence[tuple[str, float, float, float, float]]],
    classes: Sequence[str],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """AP at IoU `iou_threshold` per class (NaN for classes with no ground truth).

    predictions: image id -> detections. ground_truth: image id -> (class, x1, y1, x2, y2).
    Each ground-truth box matches at most one prediction, highest confidence first.
    """
    result: dict[str, float] = {}
    for cls in classes:
        gts = {img: [g[1:] for g in boxes if g[0] == cls] for img, boxes in ground_truth.items()}
        n_gt = sum(len(v) for v in gts.values())
        preds = sorted(
            ((img, d) for img, dets in predictions.items() for d in dets if d.class_name == cls),
            key=lambda p: p[1].confidence,
            reverse=True,
        )
        used = {img: [False] * len(v) for img, v in gts.items()}
        tp: list[bool] = []
        for img, d in preds:
            box = (d.x1, d.y1, d.x2, d.y2)
            best, best_i = 0.0, -1
            for i, g in enumerate(gts.get(img, [])):
                if used[img][i]:
                    continue
                overlap = box_iou(box, g)
                if overlap > best:
                    best, best_i = overlap, i
            hit = best_i >= 0 and best >= iou_threshold
            if hit:
                used[img][best_i] = True
            tp.append(hit)
        result[cls] = average_precision(tp, [d.confidence for _, d in preds], n_gt)
    return result


def extract_test_split(zip_path: Path = DRIVE_ZIP, dest: Path = TEST_DEST, class_names: Sequence[str] | None = None) -> Path:
    """Extract only data.yaml, images/test/ and labels/test/ from merged.zip and
    write `dest/data.yaml` pointing at them. Returns the data.yaml path.

    Requires merged.done next to the zip. Reuses an earlier extraction from the
    same zip. Fails if the zip's class names differ from `class_names`, since the
    label IDs would then mean different classes.
    """
    done = zip_path.with_suffix(".done")
    if not done.exists():
        raise RuntimeError(f"{done} is missing: {zip_path.name} was never verified. Run notebook 02 first.")
    info = json.loads(done.read_text())
    marker = dest / EXTRACTED
    data_yaml = dest / "data.yaml"
    if marker.exists() and json.loads(marker.read_text()) == info and data_yaml.exists():
        logger.info("%s already extracted from this zip, reusing it", dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m for m in zf.namelist()
                if m == "data.yaml" or m.startswith(("images/test/", "labels/test/"))
            ]
            if "data.yaml" not in members:
                raise RuntimeError(f"{zip_path} has no data.yaml at its root")
            for m in members:
                zf.extract(m, dest)
        n_images = sum(1 for m in members if m.startswith("images/test/") and not m.endswith("/"))
        logger.info("Extracted %d test images from %s in %.0fs", n_images, zip_path.name, time.time() - t0)
        marker.write_text(json.dumps(info))

    data = yaml.safe_load(data_yaml.read_text())
    zip_names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    names = list(class_names) if class_names is not None else zip_names
    if zip_names != names:
        raise RuntimeError(f"merged.zip classes {zip_names} differ from configs/classes.yaml {names}")
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dest.absolute()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }, sort_keys=False))
    return data_yaml


def load_split(data_yaml: Path, split: str = "test", max_images: int | None = None) -> list[tuple[Path, Path]]:
    """(image, label) pairs of a split in a YOLO data.yaml."""
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent))
    images_dir = root / data[split]
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_images:
        images = images[:max_images]
    labels_dir = root / str(data[split]).replace("images", "labels", 1)
    return [(img, labels_dir / img.relative_to(images_dir).with_suffix(".txt")) for img in images]


def evaluate(
    detector: DetectorBackend, pairs: Sequence[tuple[Path, Path]], class_names: Sequence[str]
) -> dict[str, float]:
    """mAP50 per class of `detector` on (image, label) pairs."""
    predictions: dict[str, list[Detection]] = {}
    ground_truth: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for i, (img_path, label_path) in enumerate(pairs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read %s", img_path)
            continue
        if isinstance(detector, HybridDetector):
            detector.reset()  # unrelated images: never reuse the previous image's boxes
        key = str(img_path)
        predictions[key] = detector.detect(frame)
        ground_truth[key] = [(class_names[c], *box) for c, *box in yolo_label_boxes(label_path) if c < len(class_names)]
        if (i + 1) % 500 == 0:
            logger.info("  %d / %d images", i + 1, len(pairs))
    return ap50_per_class(predictions, ground_truth, class_names)


# ---------------------------------------------------------------------- tables

def format_counts_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    backends = list(results)
    classes = sorted({c for r in results.values() for c in r["counts"]})
    header = ["class"] + [LABELS.get(b, b) for b in backends]
    rows = [
        ["frames"] + [str(results[b]["frames"]) for b in backends],
        ["FPS"] + [f"{results[b]['fps']:.1f}" for b in backends],
        ["mean ms"] + [f"{results[b]['mean_ms']:.0f}" for b in backends],
        ["max ms"] + [f"{results[b]['max_ms']:.0f}" for b in backends],
    ]
    rows += [
        [("* " if c in RARE_CLASSES else "  ") + c] + [str(results[b]["counts"].get(c, 0)) for b in backends]
        for c in classes
    ]
    return _table(header, rows)


def format_map_table(
    maps: Mapping[str, Mapping[str, float | None]], classes: Sequence[str]
) -> str:
    """class | <backend> mAP50 ... ; "-" = backend can't output this class, "n/a" = no test boxes."""
    backends = list(maps)
    header = ["class"] + [f"{LABELS.get(b, b)} mAP50" for b in backends]

    def cell(v: float | None) -> str:
        if v is None:
            return "-"
        return "n/a" if math.isnan(v) else f"{v:.3f}"

    rows = [[("* " if c in RARE_CLASSES else "  ") + c] + [cell(maps[b].get(c)) for b in backends] for c in classes]
    for label, subset in (("mean (all)", classes), ("mean (rare *)", [c for c in classes if c in RARE_CLASSES])):
        rows.append([label] + [f"{mean_map(maps[b], subset):.3f}" for b in backends])
    return _table(header, rows) + '\n* rare class. "-" = the model cannot detect it (0 in the means). n/a = no test boxes.'


def mean_map(per_class: Mapping[str, float | None], classes: Iterable[str]) -> float:
    """Mean over classes with test boxes; classes the model can't output count as 0."""
    values = [0.0 if per_class.get(c) is None else per_class[c] for c in classes]
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w) for i, (v, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, rows)])


# ---------------------------------------------------------------------- main

def outputs_of(detector: DetectorBackend) -> set[str]:
    """Class names a backend can output."""
    if isinstance(detector, HybridDetector):
        return outputs_of(detector.fast) | outputs_of(detector.slow)
    if hasattr(detector, "index_to_class"):
        return set(detector.index_to_class)
    allowed = getattr(detector, "allowed_names", None)
    return set(allowed) if allowed is not None else set(detector.names.values())  # type: ignore[attr-defined]


def run_eval(
    det_cfg: Mapping[str, Any], backends: Sequence[str], data_yaml: Path, class_names: Sequence[str],
    eval_conf: float, max_images: int | None,
) -> dict[str, dict[str, float | None]]:
    pairs = load_split(data_yaml, "test", max_images)
    logger.info("Evaluating on %d test images", len(pairs))
    # Low thresholds for mAP, open-vocab on every image, COCO model limited to our classes.
    overrides = {
        "conf": eval_conf,
        "class_conf": {},
        "hybrid": {**det_cfg.get("hybrid", {}), "openvocab_every": 1},
        "coco": {**det_cfg.get("coco", {}), "allowed_classes": list(class_names)},
    }
    maps: dict[str, dict[str, float | None]] = {}
    for backend in backends:
        logger.info("mAP50: %s", backend)
        detector = build_detector(det_cfg, backend, overrides=overrides)
        can = outputs_of(detector)
        ap = evaluate(detector, pairs, class_names)
        maps[backend] = {c: (ap[c] if c in can else None) for c in class_names}
    return maps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare COCO-only, open-vocab and hybrid detectors")
    parser.add_argument("sources", nargs="*", help="images, folders of images, or videos")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backends", nargs="+", default=["ultralytics", "openvocab", "hybrid"],
                        choices=list(LABELS), help="backends to compare (ultralytics = COCO-only)")
    parser.add_argument("--max-frames", type=int, default=300, help="max frames per video")
    parser.add_argument("--openvocab-every", type=int, help="hybrid: override openvocab_every")
    parser.add_argument("--eval", action="store_true", help="compute mAP50 on the merged test split")
    parser.add_argument("--zip", type=Path, default=DRIVE_ZIP, help="merged.zip (needs merged.done next to it)")
    parser.add_argument("--dest", type=Path, default=TEST_DEST, help="where the test split is extracted")
    parser.add_argument("--eval-conf", type=float, default=0.01, help="detection threshold for mAP")
    parser.add_argument("--max-images", type=int, help="evaluate only the first N test images")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    det_cfg = cfg["detector"]
    if args.openvocab_every:
        det_cfg = {**det_cfg, "hybrid": {**det_cfg.get("hybrid", {}), "openvocab_every": args.openvocab_every}}

    if args.sources:
        frames = [f for _, f in iter_frames(args.sources, args.max_frames)]
        if not frames:
            logger.error("No frames found in %s", args.sources)
            return 1
        results = {}
        for backend in args.backends:
            detector = build_detector(det_cfg, backend)
            results[backend] = benchmark(detector, frames)
        print("\nSpeed and detections per class (* = rare class)\n")
        print(format_counts_table(results))

    want_eval = args.eval or (not args.sources and args.zip.exists())
    if want_eval:
        if not (args.zip.exists() and args.zip.with_suffix(".done").exists()):
            logger.error("No verified %s (needs %s): skipping mAP", args.zip, args.zip.with_suffix(".done").name)
            return 1 if args.eval else 0
        class_names = load_class_names(ROOT / det_cfg.get("classes_file", "configs/classes.yaml"))
        data_yaml = extract_test_split(args.zip, args.dest, class_names)
        maps = run_eval(det_cfg, args.backends, data_yaml, class_names, args.eval_conf, args.max_images)
        print("\nmAP50 per class on the TEST split\n")
        print(format_map_table(maps, class_names))
    elif not args.sources:
        logger.error("Give images/videos to benchmark, or --eval with a merged.zip")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
