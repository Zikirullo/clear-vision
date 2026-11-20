"""Export a trained run to Core ML for iPhone on a Mac, and check it against the .pt.

    .venv-export/bin/python scripts/export_coreml.py --run budget_v1 --images path/to/photos
    .venv-export/bin/python scripts/export_coreml.py --run budget_v1 --images photos --test-images merged/images/test

Run it with the export environment `.venv-export` (Python 3.11, coremltools 9.0,
torch 2.7.0, numpy < 2.4; see HISTORY.md 2026-11-20). numpy 2.4+ breaks coremltools
9.0's PyTorch converter ("only 0-dimensional arrays can be converted to Python scalars",
coremltools issue #2633; the fix is only in a 9.1 pre-release).

What it does:
1. Exports models/<run>/best.pt to Core ML at the training imgsz (from metrics.json),
   FP16, with NMS embedded (nms=True) as the ultralytics_yolo Flutter plugin needs for a
   YOLO11 detect model on iOS ("nms=True embeds NMS where supported"; the official nms=False
   assets are NMS-free YOLO26 models). The NMS keeps boxes from the lowest per-class
   threshold in configs/default.yaml (0.15), so the app can still apply 0.15 to curb,
   stairs, pothole, pole and manhole. If the export fails with the numpy TypeError,
   it retries once with nms=False and says so.
2. Writes exports/<run>/mobile/clear_vision_<run>.mlpackage and .mlpackage.zip
   (the zip is what Flutter's assets/models/ takes on iOS).
3. Checks the embedded metadata (task detect, the 20 class names, imgsz).
4. Runs the .mlpackage and best.pt on the given images and prints the detections side by
   side, the share of matching boxes (same class, IoU > 0.5) and the time per image.
   Pass = at least 95% of all boxes have a partner in the other model.
5. Writes exports/<run>/mobile/coreml_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

import scripts.mobile_export as me  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PASS_SHARE = 0.95
DRIVE_HINT = "MyDrive/clear_vision/exports/{run}/best.pt"

Box = tuple[str, float, float, float, float, float]  # class, conf, x1, y1, x2, y2 (normalized)


# ---------------------------------------------------------------------- matching (pure)

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(a: Sequence[Box], b: Sequence[Box], iou: float = 0.5) -> int:
    """Number of one-to-one pairs between two models' boxes: same class, IoU > `iou`,
    most confident boxes paired first."""
    used: set[int] = set()
    pairs = 0
    for box in sorted(a, key=lambda x: -x[1]):
        best, best_j = iou, -1
        for j, other in enumerate(b):
            if j in used or other[0] != box[0]:
                continue
            overlap = box_iou(box[2:], other[2:])
            if overlap > best:
                best, best_j = overlap, j
        if best_j >= 0:
            used.add(best_j)
            pairs += 1
    return pairs


def match_share(pairs: int, n_a: int, n_b: int) -> float:
    """Share of all boxes (both models) that have a partner: 2 * pairs / (n_a + n_b).
    Counts missing and extra boxes alike; 1.0 if neither model found anything."""
    return 1.0 if n_a + n_b == 0 else 2 * pairs / (n_a + n_b)


# ---------------------------------------------------------------------- steps

def export(weights: Path, out_dir: Path, name: str, imgsz: int, conf: float, nms: bool = True,
           allow_fallback: bool = True) -> tuple[Path, dict[str, Any]]:
    """Export to Core ML FP16 from a temporary copy of the weights (Ultralytics writes the
    package next to them). Returns (the .mlpackage in out_dir, the settings that worked)."""
    from ultralytics import YOLO

    work = out_dir / "_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copyfile(weights, work / "best.pt")
    settings = {"format": "coreml", "imgsz": imgsz, "quantize": 16, "nms": nms, "conf": conf}
    attempts: list[dict[str, Any]] = []
    while True:
        args = dict(settings)
        if not settings["nms"]:
            args.pop("conf")  # without NMS there is no embedded threshold
        try:
            produced = Path(YOLO(str(work / "best.pt")).export(**args))
            break
        except TypeError as e:
            attempts.append({**settings, "error": str(e)})
            numpy_bug = "0-dimensional arrays" in str(e)
            if not (numpy_bug and settings["nms"] and allow_fallback):
                raise
            print(f"Export with nms=True failed ({e}); retrying with nms=False (last resort).", flush=True)
            settings["nms"] = False
    package = me.copy_artifact(produced, out_dir / f"{name}.mlpackage")
    shutil.rmtree(work, ignore_errors=True)
    return package, {**settings, "failed_attempts": attempts}


def list_images(folders: Sequence[Path], limit: int | None = None) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        found = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        images += found[:limit] if limit else found
    return images


def predict_boxes(model: Any, image: Path, imgsz: int, conf: float, **kwargs: Any) -> tuple[list[Box], float]:
    """(boxes at confidence >= conf, seconds) for one image."""
    start = time.perf_counter()
    r = model.predict(str(image), imgsz=imgsz, conf=conf, verbose=False, **kwargs)[0]
    seconds = time.perf_counter() - start
    boxes = [(r.names[int(c)], float(p), *map(float, b))
             for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())
             if float(p) >= conf]
    return boxes, seconds


def compare(weights: Path, package: Path, images: Sequence[Path], imgsz: int, conf: float,
            same_input: bool = True) -> dict[str, Any]:
    """Run both models on every image; print detections side by side; return the totals.

    The Core ML model has a fixed square input (imgsz x imgsz, letterboxed). With
    `same_input`, best.pt gets the same square input (rect=False) so differences come from
    the export only; otherwise it uses Ultralytics' default rectangular letterbox, which
    changes scores by itself.
    """
    from ultralytics import YOLO

    pt, cm = YOLO(str(weights)), YOLO(str(package), task="detect")
    for warm in images[:2]:  # first runs load and compile; don't time them
        pt.predict(str(warm), imgsz=imgsz, verbose=False)
        cm.predict(str(warm), imgsz=imgsz, verbose=False)
    rows, pairs_total, n_pt, n_cm, t_pt, t_cm = [], 0, 0, 0, [], []
    print(f"\n{'image':34s} {'best.pt':44s} {'Core ML':44s} match")
    for image in images:
        a, ta = predict_boxes(pt, image, imgsz, conf, rect=not same_input)
        b, tb = predict_boxes(cm, image, imgsz, conf)
        pairs = match_boxes(a, b)
        pairs_total, n_pt, n_cm = pairs_total + pairs, n_pt + len(a), n_cm + len(b)
        t_pt.append(ta)
        t_cm.append(tb)
        fmt = lambda boxes: ", ".join(f"{c} {p:.2f}" for c, p, *_ in sorted(boxes, key=lambda x: -x[1])[:3]) or "-"  # noqa: E731
        print(f"{image.name[:34]:34s} {fmt(a)[:44]:44s} {fmt(b)[:44]:44s} {pairs}/{max(len(a), len(b))}")
        rows.append({"image": image.name, "pt": a, "coreml": b, "pairs": pairs})
    share = match_share(pairs_total, n_pt, n_cm)
    return {
        "images": len(images), "conf": conf, "same_input": same_input, "boxes_pt": n_pt, "boxes_coreml": n_cm, "pairs": pairs_total,
        "match_share": share, "pass": share >= PASS_SHARE,
        "ms_per_image_pt": 1000 * statistics.median(t_pt) if t_pt else None,
        "ms_per_image_coreml": 1000 * statistics.median(t_cm) if t_cm else None,
        "per_image": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a run to Core ML (iPhone) and check it against the .pt")
    parser.add_argument("--run", required=True, help="run name, e.g. budget_v1 (models/<run>/best.pt)")
    parser.add_argument("--imgsz", type=int, help="default: models/<run>/metrics.json, else 512")
    parser.add_argument("--no-nms", action="store_true", help="export without embedded NMS")
    parser.add_argument("--images", type=Path, action="append", default=[], help="folder of check images (repeatable)")
    parser.add_argument("--test-images", type=Path, help="folder of test-split images; the first --test-limit are used")
    parser.add_argument("--test-limit", type=int, default=50)
    parser.add_argument("--compare-conf", type=float, default=0.25, help="confidence at which boxes are compared")
    parser.add_argument("--skip-export", action="store_true", help="only check an existing .mlpackage")
    parser.add_argument("--rect-pt", action="store_true",
                        help="give best.pt its default rectangular input instead of Core ML's square one")
    args = parser.parse_args(argv)

    run_dir = ROOT / "models" / args.run
    weights = run_dir / "best.pt"
    if not weights.exists():
        print(f"{weights} is missing. Download it from Google Drive {DRIVE_HINT.format(run=args.run)} and save it as "
              f"{weights.relative_to(ROOT)} (create the folder models/{args.run}/ if needed).")
        return 1
    import numpy

    if tuple(int(x) for x in numpy.__version__.split(".")[:2]) >= (2, 4):
        print(f"numpy {numpy.__version__} breaks coremltools 9.0 (issue #2633): use .venv-export (numpy < 2.4).")
        return 1

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    classes = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]
    default_thr, thresholds = me.thresholds(cfg, classes)
    imgsz = args.imgsz or me.read_imgsz(run_dir / "metrics.json")
    out_dir = ROOT / "exports" / args.run / "mobile"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"clear_vision_{args.run}"
    package = out_dir / f"{name}.mlpackage"

    settings: dict[str, Any] = {"skipped": True}
    if not args.skip_export:
        print(f"Exporting {weights.relative_to(ROOT)} to Core ML: imgsz {imgsz}, FP16, "
              f"nms={not args.no_nms}, NMS confidence {min(thresholds.values())}", flush=True)
        t0 = time.time()
        package, settings = export(weights, out_dir, name, imgsz, min(thresholds.values()), nms=not args.no_nms)
        settings["seconds"] = round(time.time() - t0, 1)
        print(f"-> {package.relative_to(ROOT)} ({me.size_mb(package):.1f} MB) in {settings['seconds']} s")
    zip_path = me.zip_mlpackage(package, out_dir / f"{name}.mlpackage.zip")
    print(f"-> {zip_path.relative_to(ROOT)} ({me.size_mb(zip_path):.1f} MB)")

    meta = me.coreml_metadata(package) or {}
    names = me.names_from_metadata(meta)
    meta_ok = meta.get("task") == "detect" and names == list(classes)
    print(f"Metadata: task={meta.get('task')} classes={'20 in order' if names == list(classes) else names} "
          f"imgsz={meta.get('imgsz')} -> {'OK' if meta_ok else 'BAD'}")

    images = list_images(args.images)
    test_images = list_images([args.test_images], args.test_limit) if args.test_images else []
    if not args.test_images:
        print("No --test-images given: the check uses only --images.")
    results = {}
    for label, imgs in (("photos", images), ("test", test_images)):
        if imgs:
            print(f"\n=== {label}: {len(imgs)} images, boxes compared at conf >= {args.compare_conf}, "
                  f"best.pt input: {'rectangular (default)' if args.rect_pt else 'square, same as Core ML'}")
            results[label] = compare(weights, package, imgs, imgsz, args.compare_conf, same_input=not args.rect_pt)
            r = results[label]
            print(f"\n{label}: {r['pairs']} pairs, {r['boxes_pt']} boxes from best.pt, {r['boxes_coreml']} from Core ML "
                  f"-> {100 * r['match_share']:.1f}% matching ({'PASS' if r['pass'] else 'FAIL'}, need "
                  f"{100 * PASS_SHARE:.0f}%). Median time per image: best.pt {r['ms_per_image_pt']:.1f} ms, "
                  f"Core ML {r['ms_per_image_coreml']:.1f} ms")

    metrics = {
        "run": args.run, "imgsz": imgsz, "package": package.name, "zip": zip_path.name,
        "size_mb": round(me.size_mb(package), 2), "zip_mb": round(me.size_mb(zip_path), 2),
        "settings": settings, "metadata": {"task": meta.get("task"), "classes_ok": names == list(classes),
                                           "imgsz": meta.get("imgsz")},
        "thresholds": {"default": default_thr, "classes": thresholds},
        "check": results,
        "environment": {"python": platform.python_version(), "macos": platform.mac_ver()[0],
                        **{p: version(p) for p in ("ultralytics", "coremltools", "torch", "numpy")}},
    }
    (out_dir / "coreml_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {(out_dir / 'coreml_metrics.json').relative_to(ROOT)}")
    if not results:
        print("RESULT: NOT CHECKED: no images given (--images / --test-images). The export itself worked.")
        return 3
    passed = meta_ok and all(r["pass"] for r in results.values())
    print("RESULT: " + ("PASS" if passed else "FAIL") +
          f". Upload {package.name}, {zip_path.name} and coreml_metrics.json to Google Drive "
          f"MyDrive/clear_vision/exports/{args.run}/mobile/ next to the TFLite files.")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
