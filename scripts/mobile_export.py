"""Helpers for notebooks/07_export_mobile.ipynb: export a trained run for phones.

Kept out of the notebook so they are unit-tested (tests/test_mobile_export.py).

Exports (Ultralytics >= 8.4.142; in 8.4.207 TFLite is format="litert", precision is `quantize`):
- TFLite INT8:  format="litert", quantize=8, calibrated on CALIB_IMAGES train images.
- TFLite FP16:  format="saved_model" (onnx2tf), which also writes <name>_float16.tflite.
                LiteRT export has no FP16 option.
- Core ML FP16: format="coreml", nms=True, quantize=16.
- Flutter plugin (ultralytics_yolo, doc/models.md):
    Android: format="litert", nms=None (NMS runs in the plugin), FP32 -> clear_vision_<RUN>.tflite
    iOS:     the Core ML export -> clear_vision_<RUN>.mlpackage and .mlpackage.zip (for assets/models/)
  The plugin reads task and class names from the metadata Ultralytics embeds
  (TFLite: appended zip with metadata.json; Core ML: user-defined metadata).
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CALIB_IMAGES = 200
DEFAULT_IMGSZ = 512
CRITICAL_CLASSES = ("curb", "stairs", "pothole", "pole", "manhole")


def read_imgsz(metrics_json: Path, default: int = DEFAULT_IMGSZ) -> int:
    """Training image size from notebook 05's metrics.json (settings.imgsz), else `default`."""
    if metrics_json.exists():
        imgsz = json.loads(metrics_json.read_text()).get("settings", {}).get("imgsz")
        if imgsz:
            return int(imgsz)
    return default


def calibration_yaml(dataset: Path, classes: Sequence[str], n: int = CALIB_IMAGES, seed: int = 42) -> Path:
    """data yaml whose val split is `n` random train images (INT8 calibration reads the val split).

    Writes calib_train.txt ("./images/train/<name>" lines) and data_calib.yaml into `dataset`.
    """
    images = sorted(p.name for p in (dataset / "images" / "train").iterdir() if p.is_file())
    if not images:
        raise RuntimeError(f"No train images in {dataset / 'images' / 'train'}")
    picks = sorted(random.Random(seed).sample(images, min(n, len(images))))
    (dataset / "calib_train.txt").write_text("".join(f"./images/train/{name}\n" for name in picks))
    out = dataset / "data_calib.yaml"
    out.write_text(yaml.safe_dump({
        "path": str(dataset.absolute()), "train": "calib_train.txt", "val": "calib_train.txt",
        "names": dict(enumerate(classes)),
    }, sort_keys=False))
    return out


def thresholds(cfg: Mapping[str, Any], classes: Sequence[str]) -> tuple[float, dict[str, float]]:
    """(default, per-class) detection thresholds of the fine-tuned model from
    configs/default.yaml (detector.finetuned.conf and .class_conf)."""
    ft = cfg.get("detector", {}).get("finetuned", {}) or {}
    default = float(ft.get("conf", 0.25))
    per_class = ft.get("class_conf") or {}
    return default, {c: float(per_class.get(c, default)) for c in classes}


def write_side_files(
    dest: Path, classes: Sequence[str], default: float, thr: Mapping[str, float], run: str, imgsz: int,
) -> None:
    """labels.txt (class names in ID order) and thresholds.json next to the models."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "labels.txt").write_text("".join(f"{c}\n" for c in classes))
    (dest / "thresholds.json").write_text(json.dumps({
        "run": run, "imgsz": imgsz, "default": default, "classes": dict(thr),
        "note": "minimum confidence per class (configs/default.yaml detector.finetuned)",
    }, indent=2))


def zip_mlpackage(mlpackage: Path, dest_zip: Path) -> Path:
    """Zip an .mlpackage folder so the package folder is the top entry (plugin assets/models/)."""
    if not mlpackage.is_dir():
        raise FileNotFoundError(f"{mlpackage} is not an .mlpackage folder")
    dest_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(mlpackage.rglob("*")):
            if path.is_file():
                zf.write(path, (Path(mlpackage.name) / path.relative_to(mlpackage)).as_posix())
    return dest_zip


def tflite_metadata(path: Path) -> dict[str, Any] | None:
    """Ultralytics' metadata.json appended to a .tflite as a zip; None if missing."""
    try:
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("metadata.json"))
    except (zipfile.BadZipFile, KeyError):
        return None


def coreml_metadata(path: Path) -> dict[str, str] | None:
    """User-defined metadata of an .mlpackage (needs coremltools; reads the spec only,
    so it works on Linux too). None if coremltools is missing."""
    try:
        import coremltools as ct
    except ImportError:
        return None
    model = ct.models.MLModel(str(path), skip_model_load=True)
    return dict(model.user_defined_metadata)


def names_from_metadata(meta: Mapping[str, Any] | None) -> list[str] | None:
    """Class names from either metadata flavour (dict, or a str(dict) as Core ML stores it)."""
    if not meta or "names" not in meta:
        return None
    names = meta["names"]
    if isinstance(names, str):
        import ast

        names = ast.literal_eval(names)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda k: int(k))]
    return list(names)


def size_mb(path: Path) -> float:
    """Size of a file or a folder (.mlpackage, saved_model) in MB."""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


def copy_artifact(src: Path, dest: Path) -> Path:
    """Copy a file or folder to `dest`, replacing what was there."""
    if dest.is_dir():
        shutil.rmtree(dest)
    elif dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copyfile(src, dest)
    return dest


def result_row(name: str, path: Path, recall: Mapping[str, float | None] | None,
               map50: float | None, base_map50: float | None, base_mb: float) -> dict[str, Any]:
    """One row of the comparison table: mAP50, its loss vs the .pt, recall of the critical
    classes (at their own thresholds), size and size ratio. `recall` / `map50` None = not
    validated here (Core ML outside macOS)."""
    row: dict[str, Any] = {"model": name, "file": path.name, "mAP50": map50}
    row["mAP50 loss %"] = (None if map50 is None or not base_map50
                           else round(100 * (base_map50 - map50) / base_map50, 2))
    for c in CRITICAL_CLASSES:
        row[f"recall {c}"] = None if recall is None else recall.get(c)
    mb = size_mb(path)
    row["size MB"] = round(mb, 2)
    row["x .pt size"] = round(mb / base_mb, 2) if base_mb else None
    return row


def format_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = list(rows[0])

    def cell(v: Any) -> str:
        return "-" if v is None else f"{v:.3f}" if isinstance(v, float) and not isinstance(v, bool) else str(v)

    body = [[cell(r[k]) for k in keys] for r in rows]
    widths = [max(len(k), *(len(b[i]) for b in body)) for i, k in enumerate(keys)]
    fmt = lambda r: "  ".join(v.ljust(w) if i < 2 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([fmt(keys), "  ".join("-" * w for w in widths), *map(fmt, body)])


# ---------------------------------------------------------------------- uniform evaluation

def letterbox(image: Any, size: int) -> tuple[Any, float, int, int]:
    """Resize keeping the aspect ratio and pad to size x size with gray (114), like Ultralytics.
    Returns (RGB uint8 canvas, scale, pad_x, pad_y)."""
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas[..., ::-1].copy(), scale, pad_x, pad_y  # BGR -> RGB


def decode_yolo(raw: Any, size: int, conf: float, iou: float, max_det: int = 300) -> tuple[Any, Any, Any]:
    """Raw YOLO detect output (4 + nc rows by anchors, either orientation; box xywh in pixels or
    normalized) -> (boxes xyxy in letterbox pixels, scores, class ids) after per-class NMS."""
    import numpy as np
    import torch
    import torchvision

    y = np.asarray(raw, dtype=np.float32)
    y = y[0] if y.ndim == 3 else y
    if y.shape[0] > y.shape[1]:  # (anchors, 4 + nc) -> (4 + nc, anchors)
        y = y.T
    boxes, scores_all = y[:4], y[4:]
    if np.abs(boxes).max() <= 2.0:  # normalized (litert-torch exports) -> letterbox pixels
        boxes = boxes * size
    cls = scores_all.argmax(0)
    scores = scores_all[cls, np.arange(scores_all.shape[1])]
    keep = scores >= conf
    cx, cy, w, h = boxes[:, keep]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    scores, cls = scores[keep], cls[keep]
    if len(scores) == 0:
        return xyxy, scores, cls
    idx = torchvision.ops.batched_nms(torch.from_numpy(xyxy), torch.from_numpy(scores), torch.from_numpy(cls), iou)
    idx = idx[:max_det].numpy()
    return xyxy[idx], scores[idx], cls[idx]


class TFLiteDetector:
    """Runs any of our .tflite exports with the LiteRT interpreter and returns Detections.

    Reads the layout from the model itself: NHWC (TensorFlow route) or NCHW (litert-torch),
    float or int8 input, pixel or normalized boxes. Ultralytics 8.4.207's own LiteRT reader
    assumes normalized boxes and scores the TensorFlow-route FP16 file at 0, so all .tflite
    files are evaluated through this one reader instead.
    """

    def __init__(self, path: Path, classes: Sequence[str], conf: float = 0.001, iou: float = 0.7) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # older runtimes
            from tensorflow.lite import Interpreter  # type: ignore[no-redef]
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.nhwc = int(self.inp["shape"][-1]) == 3
        self.size = int(self.inp["shape"][1] if self.nhwc else self.inp["shape"][2])
        self.classes, self.conf, self.iou = list(classes), conf, iou
        self.device = "cpu"
        self.last_inference_ms = 0.0

    def detect(self, frame: Any) -> list[Any]:
        import time

        import numpy as np

        from src.detection import Detection

        canvas, scale, px, py = letterbox(frame, self.size)
        x = canvas.astype(np.float32) / 255.0
        x = x[None] if self.nhwc else x.transpose(2, 0, 1)[None]
        if self.inp["dtype"] in (np.int8, np.uint8):
            s, z = self.inp["quantization"]
            x = (x / s + z).round().clip(np.iinfo(self.inp["dtype"]).min, np.iinfo(self.inp["dtype"]).max)
        start = time.perf_counter()
        self.interpreter.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out["index"])
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        if self.out["dtype"] in (np.int8, np.uint8):
            s, z = self.out["quantization"]
            raw = (raw.astype(np.float32) - z) * s
        boxes, scores, cls = decode_yolo(raw, self.size, self.conf, self.iou)
        h, w = frame.shape[:2]
        out = []
        for (x1, y1, x2, y2), p, c in zip(boxes, scores, cls):
            x1, x2 = ((np.array([x1, x2]) - px) / scale / w).clip(0, 1)
            y1, y2 = ((np.array([y1, y2]) - py) / scale / h).clip(0, 1)
            if x2 > x1 and y2 > y1:
                out.append(Detection(self.classes[int(c)], float(p), float(x1), float(y1), float(x2), float(y2)))
        return out


def recall_at_thresholds(
    predictions: Mapping[str, Sequence[Any]], ground_truth: Mapping[str, Sequence[tuple]],
    thresholds: Mapping[str, float], classes: Sequence[str], iou: float = 0.5,
) -> dict[str, float | None]:
    """Share of ground-truth objects found per class at that class's own confidence threshold
    (what the app would actually report). None for classes without test objects."""
    from scripts.train_utils import find_misses

    out: dict[str, float | None] = {}
    for c in classes:
        total = sum(1 for boxes in ground_truth.values() for b in boxes if b[0] == c)
        if total == 0:
            out[c] = None
            continue
        missed = len(find_misses(predictions, ground_truth, [c], conf=thresholds.get(c, 0.25), iou=iou))
        out[c] = (total - missed) / total
    return out


def evaluate_model(
    detector: Any, pairs: Sequence[tuple[Path, Path]], classes: Sequence[str], thresholds: Mapping[str, float],
) -> tuple[float, dict[str, float], dict[str, float | None]]:
    """(mAP50, AP50 per class, recall per class at its threshold) with notebook 08's evaluator."""
    import math

    from scripts.compare_detectors import ap50_per_class
    from scripts.train_utils import predict_split

    preds, gts = predict_split(detector, pairs, classes)
    ap = ap50_per_class(preds, gts, classes)
    present = [v for v in ap.values() if not math.isnan(v)]
    return (sum(present) / len(present) if present else float("nan")), ap, \
        recall_at_thresholds(preds, gts, thresholds, classes)


# ---------------------------------------------------------------------- command line (one process per step)

def export_one(weights: Path, fmt: str, out: Path, imgsz: int, **kwargs: Any) -> Path:
    """Export `weights` with Ultralytics and move the artifact to `out`.

    For format="saved_model" the FP16 file inside the SavedModel folder is the artifact.
    Run in a fresh process (see main): exports install packages (numpy, torch versions)
    that break modules already loaded in the same process.
    """
    from ultralytics import YOLO

    produced = Path(YOLO(str(weights)).export(format=fmt, imgsz=imgsz, **kwargs))
    if fmt == "saved_model":
        produced = next(produced.glob("*_float16.tflite"))
    return copy_artifact(produced, out)


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)  # "8" -> 8, "true" -> True, "null" -> None, "0.15" -> 0.15


def main(argv: Sequence[str] | None = None) -> int:
    """`export WEIGHTS FORMAT OUT [key=value ...] --imgsz N` or
    `evaluate MODEL DATA_YAML --imgsz N --thresholds thresholds.json`.
    The last stdout line is `RESULT <json>`."""
    import argparse

    parser = argparse.ArgumentParser(description="Export or evaluate one phone model (one process per step)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("weights", type=Path)
    ex.add_argument("format")
    ex.add_argument("out", type=Path)
    ex.add_argument("--imgsz", type=int, required=True)
    ex.add_argument("options", nargs="*", help="Ultralytics export args as key=value, e.g. quantize=8 nms=true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("model", type=Path)
    ev.add_argument("data", type=Path)
    ev.add_argument("--imgsz", type=int, required=True)
    ev.add_argument("--thresholds", type=Path, required=True)
    ev.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    if args.cmd == "export":
        options = {k: _parse_value(v) for k, v in (o.split("=", 1) for o in args.options)}
        path = export_one(args.weights, args.format, args.out, args.imgsz, **options)
        print("RESULT " + json.dumps({"path": str(path), "size_mb": round(size_mb(path), 2)}))
        return 0

    import platform

    if args.model.suffix == ".mlpackage" and platform.system() != "Darwin":  # before importing torch
        print("RESULT " + json.dumps({"skipped": "Core ML runs only on macOS"}))
        return 0

    from scripts.compare_detectors import load_split
    from src.detector import Detector

    data = yaml.safe_load(args.data.read_text())
    names = data["names"]
    classes = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    thr = json.loads(args.thresholds.read_text())["classes"]
    detector = (TFLiteDetector(args.model, classes) if args.model.suffix == ".tflite"
                else Detector(args.model, conf=0.001, imgsz=args.imgsz, warm_up=False, class_names=classes))
    pairs = load_split(args.data, args.split)
    m50, ap, recall = evaluate_model(detector, pairs, classes, thr)
    print("RESULT " + json.dumps({"map50": m50, "ap50": ap, "recall": recall, "images": len(pairs)}))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
