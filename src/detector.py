"""Object detection backends.

Every backend returns plain `Detection` objects with coordinates normalized to
0-1 and class names from configs/classes.yaml (lowercase, underscores), so the
tracker and navigator never depend on frame resolution, Ultralytics types or
which model produced a box.

Backends (`detector.backend` in configs/default.yaml, or `--backend`):

- ultralytics: a closed-set COCO YOLO model (.pt) via Ultralytics.
- onnx:        the same COCO model exported to ONNX, run by ONNX Runtime.
- openvocab:   YOLOE (fallback YOLO-World) prompted with text phrases for our
               classes. No fine-tuning needed. The text encoder runs once at
               the first start; the prompted model is cached in
               `models/openvocab/` (.pt, or .onnx) and later starts load it
               without the text encoder.
- hybrid:      the fast COCO model every frame for COCO classes, plus the
               open-vocab model every Nth frame for the rest, merged with NMS.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.detection import Detection

__all__ = [
    "BACKENDS",
    "Detection",
    "Detector",
    "DetectorBackend",
    "HybridDetector",
    "OpenVocabDetector",
    "build_detector",
    "canonical_name",
    "merge_detections",
    "select_device",
]

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("ultralytics", "onnx", "openvocab", "hybrid")

DEFAULT_MODEL_PATH = "models/yolo11n.pt"
DEFAULT_OPENVOCAB_MODEL = "models/yoloe-11s-seg.pt"
DEFAULT_OPENVOCAB_FALLBACK = "models/yolov8s-worldv2.pt"
DEFAULT_CONF = 0.4
DEFAULT_IMGSZ = 640

MIN_MACOS_FOR_MPS = 15


# ---------------------------------------------------------------------- helpers

def _macos_major() -> int:
    version = platform.mac_ver()[0]
    return int(version.split(".")[0]) if version else 0


def select_device(preference: str = "auto") -> str:
    """Return "cuda" if available, "mps" on Apple Silicon when usable, else "cpu".

    Recent PyTorch MPS kernels call macOS 15 APIs; on macOS 14 they abort the
    whole process with a native exception that Python cannot catch, so "auto"
    only picks MPS on macOS 15+.
    """
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and _macos_major() >= MIN_MACOS_FOR_MPS:
        return "mps"
    return "cpu"


def canonical_name(name: str) -> str:
    """Our naming: lowercase with underscores ("fire hydrant" -> "fire_hydrant")."""
    return name.strip().lower().replace(" ", "_")


def resolve_path(path: str | Path, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_class_names(path: str | Path) -> list[str]:
    """Class names from configs/classes.yaml, in class-ID order."""
    with Path(path).open() as f:
        return [canonical_name(n) for n in yaml.safe_load(f)["names"]]


def load_prompts(path: str | Path) -> dict[str, list[str]]:
    """Class -> text phrases from configs/prompts.yaml. A value may be a string or a list."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)["prompts"]
    return {canonical_name(k): [v] if isinstance(v, str) else list(v) for k, v in raw.items()}


def threshold_for(class_name: str, class_conf: Mapping[str, float], default: float) -> float:
    return float(class_conf.get(class_name, default))


def filter_by_conf(
    detections: Iterable[Detection], class_conf: Mapping[str, float], default: float
) -> list[Detection]:
    """Keep detections at or above their class's threshold (`default` if not listed)."""
    return [d for d in detections if d.confidence >= threshold_for(d.class_name, class_conf, default)]


def iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(
    detections: Sequence[Detection],
    iou_threshold: float,
    class_agnostic: bool = False,
    scores: Sequence[float] | None = None,
) -> list[Detection]:
    """Greedy non-maximum suppression. `scores` (default: confidence) ranks boxes."""
    if scores is None:
        scores = [d.confidence for d in detections]
    order = sorted(range(len(detections)), key=lambda i: scores[i], reverse=True)
    kept: list[Detection] = []
    for i in order:
        d = detections[i]
        if all(
            iou(d, k) <= iou_threshold
            for k in kept
            if class_agnostic or k.class_name == d.class_name
        ):
            kept.append(d)
    return kept


def merge_detections(
    groups: Iterable[Sequence[Detection]],
    iou_threshold: float,
    agnostic_iou: float | None,
    class_conf: Mapping[str, float],
    default_conf: float,
) -> list[Detection]:
    """Merge detections from several models.

    1. Per-class NMS at `iou_threshold` removes the same object found twice.
    2. If `agnostic_iou` is set, overlapping boxes with different classes
       (e.g. "scooter" from one model and "motorcycle" from the other) keep
       only one. Models score differently, so the box with the higher
       confidence relative to its class threshold wins.
    """
    merged = nms([d for g in groups for d in g], iou_threshold)
    if agnostic_iou is None:
        return merged
    relative = [d.confidence / max(threshold_for(d.class_name, class_conf, default_conf), 1e-6) for d in merged]
    return nms(merged, agnostic_iou, class_agnostic=True, scores=relative)


def result_to_detections(result: Any, index_to_class: Mapping[int, str] | Sequence[str]) -> list[Detection]:
    """Ultralytics Results -> Detections (boxes only; segmentation masks are ignored)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxyn = boxes.xyxyn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    return [
        Detection(
            class_name=index_to_class[int(c)],
            confidence=float(p),
            x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
        )
        for b, p, c in zip(xyxyn, confs, cls_ids)
    ]


class DetectorBackend(Protocol):
    """What main.py needs from any backend."""

    device: str
    last_inference_ms: float

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def _warm_up(backend: DetectorBackend, imgsz: int | Sequence[int]) -> None:
    # The first passes are slow (weight transfer, kernel compilation); do them
    # now so the live loop starts at full speed.
    side = imgsz if isinstance(imgsz, int) else max(imgsz)
    dummy = np.zeros((side, side, 3), dtype=np.uint8)
    for _ in range(2):
        backend.detect(dummy)


# ---------------------------------------------------------------------- closed-set COCO

class Detector:
    """Closed-set YOLO detector (.pt or exported .onnx) filtered to navigation classes.

    Args:
        model_path: Path to YOLO weights (.pt or .onnx). Official model names
            (e.g. yolo11n.pt) are downloaded into that path's folder if missing.
        conf: Confidence threshold for classes not in `class_conf`.
        allowed_classes: Class names to keep (any spelling, "fire hydrant" or
            "fire_hydrant"); None keeps every class.
        imgsz: Inference size in pixels.
        device: "auto", "cuda", "mps" or "cpu".
        cpu_threads: Torch CPU threads; None keeps the torch default.
        class_conf: Per-class confidence thresholds.
        warm_up: Run two dummy frames at startup.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        conf: float = DEFAULT_CONF,
        allowed_classes: Iterable[str] | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "auto",
        cpu_threads: int | None = None,
        class_conf: Mapping[str, float] | None = None,
        warm_up: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.class_conf = {canonical_name(k): float(v) for k, v in (class_conf or {}).items()}
        self.imgsz = imgsz
        self.device = select_device(device)
        self.last_inference_ms: float = 0.0
        if cpu_threads:
            torch.set_num_threads(cpu_threads)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(self.model_path))
        self.names: dict[int, str] = {i: canonical_name(n) for i, n in dict(self.model.names).items()}
        self.class_ids = self._resolve_class_ids(allowed_classes)
        self.allowed_names = (
            None if self.class_ids is None else {self.names[i] for i in self.class_ids}
        )
        # predict() takes one threshold: use the lowest, then filter per class.
        relevant = self.allowed_names if self.allowed_names is not None else set(self.names.values())
        self.predict_conf = min([conf, *(self.class_conf.get(n, conf) for n in relevant)])

        logger.info(
            "Detector ready: model=%s device=%s threads=%d conf=%.2f imgsz=%d classes=%s",
            self.model_path, self.device, torch.get_num_threads(), self.conf, self.imgsz,
            "all" if self.class_ids is None else len(self.class_ids),
        )
        if warm_up:
            _warm_up(self, self.imgsz)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Detector:
        """Build from a flat dict: model_path, conf, allowed_classes, imgsz, device, cpu_threads, class_conf."""
        return cls(
            model_path=cfg.get("model_path", DEFAULT_MODEL_PATH),
            conf=cfg.get("conf", DEFAULT_CONF),
            allowed_classes=cfg.get("allowed_classes"),
            imgsz=cfg.get("imgsz", DEFAULT_IMGSZ),
            device=cfg.get("device", "auto"),
            cpu_threads=cfg.get("cpu_threads"),
            class_conf=cfg.get("class_conf"),
        )

    def _resolve_class_ids(self, allowed: Iterable[str] | None) -> list[int] | None:
        if allowed is None:
            return None
        by_name = {name: idx for idx, name in self.names.items()}
        ids: list[int] = []
        for name in allowed:
            key = canonical_name(name)
            if key in by_name:
                ids.append(by_name[key])
            else:
                logger.warning("Unknown class %r is not in the model; ignoring it", name)
        return ids

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame and return allowed-class detections."""
        start = time.perf_counter()
        result = self.model.predict(
            frame,
            conf=self.predict_conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )[0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000

        detections = result_to_detections(result, self.names)
        # Some backends (e.g. exported ONNX models) ignore `classes=`, so filter
        # again here rather than trusting predict().
        if self.allowed_names is not None:
            detections = [d for d in detections if d.class_name in self.allowed_names]
        return filter_by_conf(detections, self.class_conf, self.conf)


def ensure_onnx(pt_path: Path, onnx_path: Path, imgsz: int) -> Path:
    """Export `pt_path` to `onnx_path` once (fixed input size `imgsz`)."""
    if onnx_path.exists():
        return onnx_path
    logger.info("Exporting %s to ONNX (imgsz=%d), one time only", pt_path, imgsz)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(pt_path)).export(format="onnx", imgsz=imgsz, simplify=True, verbose=False))
    if exported.resolve() != onnx_path.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(onnx_path)
    return onnx_path


# ---------------------------------------------------------------------- open vocabulary

def build_vocabulary(
    classes: Sequence[str], prompts: Mapping[str, Sequence[str]]
) -> tuple[list[str], list[str]]:
    """Return (phrases, index_to_class): phrase i is predicted as class index_to_class[i].

    A class may have several phrases ("stairs", "staircase"); all map back to it.
    Classes without a prompt use their own name with spaces.
    """
    phrases: list[str] = []
    index_to_class: list[str] = []
    for cls in classes:
        options = prompts.get(cls)
        if not options:
            logger.warning("No prompt for class %r in prompts.yaml; using its name", cls)
            options = [cls.replace("_", " ")]
        for phrase in options:
            if phrase in phrases:
                raise ValueError(f"Prompt {phrase!r} is used for two classes; prompts must be unique")
            phrases.append(phrase)
            index_to_class.append(cls)
    return phrases, index_to_class


def vocabulary_key(base_model: str, phrases: Sequence[str], index_to_class: Sequence[str]) -> str:
    """Short hash that changes whenever the base model, phrases or mapping change."""
    blob = json.dumps({"model": base_model, "phrases": list(phrases), "classes": list(index_to_class)})
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _model_classes() -> tuple[Any, Any]:
    """(YOLOE, YOLOWorld) from Ultralytics; YOLOE is None in versions that lack it."""
    try:
        from ultralytics import YOLOE
    except ImportError:
        YOLOE = None  # noqa: N806
    from ultralytics import YOLOWorld
    return YOLOE, YOLOWorld


def _is_world(path: Path) -> bool:
    return "world" in path.stem.lower()


def _load_prompted(path: Path) -> Any:
    """Load a cached prompted model (.pt or .onnx). No text encoder is needed."""
    if path.suffix == ".onnx":
        return YOLO(str(path))
    yoloe_cls, world_cls = _model_classes()
    if _is_world(path) or yoloe_cls is None:
        return world_cls(str(path))
    return yoloe_cls(str(path))


def _prompt_model(base: Path, phrases: list[str]) -> Any:
    """Load the base open-vocab model and run set_classes (runs the text encoder once)."""
    yoloe_cls, world_cls = _model_classes()
    if not _is_world(base) and yoloe_cls is None:
        raise ImportError(f"This Ultralytics version has no YOLOE, cannot load {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    # The text encoders (mobileclip_blt.ts, CLIP) download into the working
    # directory, so build inside models/ to keep them out of the repo root.
    with contextlib.chdir(base.parent):
        model = world_cls(base.name) if _is_world(base) else yoloe_cls(base.name)
        model.set_classes(phrases)
    return model


def load_openvocab_model(
    phrases: list[str],
    index_to_class: list[str],
    model_path: Path,
    fallback_model_path: Path | None,
    cache_dir: Path,
    runtime: str = "pt",
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[Any, Path]:
    """Return (prompted model, path it was loaded from).

    Looks for a cached prompted model for this exact vocabulary first. If there
    is none, prompts YOLOE (falling back to YOLO-World if YOLOE is unavailable
    or fails), saves the result to the cache, and, for runtime "onnx", exports
    it to ONNX. Raises if no model can be built.
    """
    if runtime not in ("pt", "onnx"):
        raise ValueError(f"openvocab runtime must be 'pt' or 'onnx', not {runtime!r}")
    candidates = [model_path] + ([fallback_model_path] if fallback_model_path else [])
    errors: list[str] = []
    for base in candidates:
        key = vocabulary_key(base.name, phrases, index_to_class)
        pt_cache = cache_dir / f"{base.stem}-{key}.pt"
        onnx_cache = cache_dir / f"{base.stem}-{key}-{imgsz}.onnx"
        wanted = onnx_cache if runtime == "onnx" else pt_cache
        if wanted.exists():
            logger.info("Loading cached open-vocab model %s", wanted)
            return _load_prompted(wanted), wanted
        try:
            if pt_cache.exists():
                model = _load_prompted(pt_cache)
            else:
                logger.info("Prompting %s with %d phrases (one time only, loads the text encoder)",
                            base.name, len(phrases))
                t0 = time.perf_counter()
                model = _prompt_model(base, phrases)
                cache_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(pt_cache))
                logger.info("Saved prompted model to %s (%.1fs)", pt_cache, time.perf_counter() - t0)
            if runtime == "onnx":
                export_openvocab_onnx(model, onnx_cache, imgsz)
                return _load_prompted(onnx_cache), onnx_cache
            return model, pt_cache
        except Exception as exc:  # noqa: BLE001 - any failure means "try the fallback"
            logger.warning("Open-vocab model %s failed: %s", base.name, exc)
            errors.append(f"{base.name}: {exc}")
    raise RuntimeError("No open-vocab model could be loaded: " + "; ".join(errors))


def export_openvocab_onnx(model: Any, onnx_path: Path, imgsz: int = DEFAULT_IMGSZ) -> Path:
    """Export a prompted model to ONNX. The class embeddings are baked in, so the
    ONNX model runs without the text encoder."""
    logger.info("Exporting prompted open-vocab model to %s (imgsz=%d)", onnx_path, imgsz)
    exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, verbose=False))
    if exported.resolve() != onnx_path.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(onnx_path)
    return onnx_path


class OpenVocabDetector:
    """Zero-training detector: YOLOE (or YOLO-World) prompted with our class phrases.

    Args:
        classes: Our class names to detect, e.g. ["pole", "stairs"].
        prompts: Class -> phrases ("traffic_cone" -> ["traffic cone"]).
        model_path: Base YOLOE weights.
        fallback_model_path: YOLO-World weights, used if YOLOE is unavailable.
        cache_dir: Where prompted models are cached.
        runtime: "pt" (prompted checkpoint) or "onnx" (exported).
        imgsz, conf, class_conf, device, cpu_threads: as for `Detector`.
        model: An already prompted model (tests inject a fake here); it must
            predict class i as phrase i of the vocabulary.
        warm_up: Run two dummy frames at startup.
    """

    def __init__(
        self,
        classes: Sequence[str],
        prompts: Mapping[str, Sequence[str]],
        model_path: str | Path = DEFAULT_OPENVOCAB_MODEL,
        fallback_model_path: str | Path | None = DEFAULT_OPENVOCAB_FALLBACK,
        cache_dir: str | Path = "models/openvocab",
        runtime: str = "pt",
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        class_conf: Mapping[str, float] | None = None,
        device: str = "auto",
        cpu_threads: int | None = None,
        model: Any | None = None,
        warm_up: bool = True,
    ) -> None:
        self.classes = [canonical_name(c) for c in classes]
        self.phrases, self.index_to_class = build_vocabulary(self.classes, prompts)
        self.conf = conf
        self.class_conf = {canonical_name(k): float(v) for k, v in (class_conf or {}).items()}
        self.predict_conf = min([conf, *(self.class_conf.get(c, conf) for c in self.classes)])
        self.imgsz = imgsz
        self.device = select_device(device)
        self.last_inference_ms: float = 0.0
        if cpu_threads:
            torch.set_num_threads(cpu_threads)

        if model is None:
            model, self.source = load_openvocab_model(
                self.phrases, self.index_to_class, Path(model_path),
                Path(fallback_model_path) if fallback_model_path else None,
                Path(cache_dir), runtime, imgsz,
            )
        else:
            self.source = Path("<injected>")
        self.model = model
        n_model = len(model.names)
        if n_model != len(self.phrases):
            raise RuntimeError(
                f"Open-vocab model has {n_model} classes but the vocabulary has {len(self.phrases)} phrases"
            )
        logger.info(
            "Open-vocab detector ready: %s device=%s imgsz=%d classes=%s",
            self.source, self.device, imgsz, ", ".join(self.classes),
        )
        if warm_up:
            _warm_up(self, imgsz)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        result = self.model.predict(
            frame, conf=self.predict_conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        detections = result_to_detections(result, self.index_to_class)
        # Several phrases may map to one class: keep the best box per object.
        if len(set(self.index_to_class)) < len(self.index_to_class):
            detections = nms(detections, 0.5)
        return filter_by_conf(detections, self.class_conf, self.conf)


# ---------------------------------------------------------------------- hybrid

class HybridDetector:
    """Fast COCO model every frame + open-vocab model every `openvocab_every` frames.

    Between open-vocab runs the last open-vocab detections are reused (one or
    a few frames old), so a stairs warning doesn't flicker on and off: when
    unsure, warn.
    """

    def __init__(
        self,
        fast: DetectorBackend,
        slow: DetectorBackend,
        openvocab_every: int = 2,
        nms_iou: float = 0.5,
        agnostic_iou: float | None = 0.7,
        class_conf: Mapping[str, float] | None = None,
        conf: float = DEFAULT_CONF,
    ) -> None:
        if openvocab_every < 1:
            raise ValueError("openvocab_every must be >= 1")
        self.fast = fast
        self.slow = slow
        self.openvocab_every = openvocab_every
        self.nms_iou = nms_iou
        self.agnostic_iou = agnostic_iou
        self.class_conf = {canonical_name(k): float(v) for k, v in (class_conf or {}).items()}
        self.conf = conf
        self.device = fast.device
        self.last_inference_ms: float = 0.0
        self.openvocab_ran = False
        self._frame = 0
        self._last_slow: list[Detection] = []

    def reset(self) -> None:
        """Forget reused open-vocab results (e.g. between unrelated images)."""
        self._frame = 0
        self._last_slow = []

    def detect(self, frame: np.ndarray) -> list[Detection]:
        fast = self.fast.detect(frame)
        total = self.fast.last_inference_ms
        self.openvocab_ran = self._frame % self.openvocab_every == 0
        if self.openvocab_ran:
            self._last_slow = self.slow.detect(frame)
            total += self.slow.last_inference_ms
        self._frame += 1
        start = time.perf_counter()
        merged = merge_detections([fast, self._last_slow], self.nms_iou, self.agnostic_iou, self.class_conf, self.conf)
        self.last_inference_ms = total + (time.perf_counter() - start) * 1000
        return merged


# ---------------------------------------------------------------------- factory

def build_detector(
    cfg: Mapping[str, Any],
    backend: str | None = None,
    root: Path = ROOT,
    overrides: Mapping[str, Any] | None = None,
) -> DetectorBackend:
    """Build the backend named by `backend` (default: cfg["backend"]) from the
    `detector:` section of configs/default.yaml.

    `overrides` replaces top-level keys of `cfg` (scripts use it to lower the
    thresholds for mAP evaluation or to run the open-vocab model every frame).
    """
    cfg = {**cfg, **(overrides or {})}
    backend = backend or cfg.get("backend", "ultralytics")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown detector backend {backend!r}; choose one of {', '.join(BACKENDS)}")

    device = cfg.get("device", "auto")
    threads = cfg.get("cpu_threads")
    conf = cfg.get("conf", DEFAULT_CONF)
    class_conf = cfg.get("class_conf") or {}
    coco_cfg = cfg.get("coco", {})
    ov_cfg = cfg.get("openvocab", {})

    def coco(allowed: Iterable[str] | None, onnx: bool) -> Detector:
        pt = resolve_path(coco_cfg.get("model_path", DEFAULT_MODEL_PATH), root)
        imgsz = coco_cfg.get("imgsz", DEFAULT_IMGSZ)
        path = pt
        if onnx:
            path = ensure_onnx(pt, resolve_path(coco_cfg.get("onnx_path", pt.with_suffix(".onnx")), root), imgsz)
        return Detector(path, conf, allowed, imgsz, device, threads, class_conf)

    def openvocab(classes: Sequence[str]) -> OpenVocabDetector:
        fallback = ov_cfg.get("fallback_model_path", DEFAULT_OPENVOCAB_FALLBACK)
        return OpenVocabDetector(
            classes,
            load_prompts(resolve_path(cfg.get("prompts_file", "configs/prompts.yaml"), root)),
            model_path=resolve_path(ov_cfg.get("model_path", DEFAULT_OPENVOCAB_MODEL), root),
            fallback_model_path=resolve_path(fallback, root) if fallback else None,
            cache_dir=resolve_path(ov_cfg.get("cache_dir", "models/openvocab"), root),
            runtime=ov_cfg.get("runtime", "pt"),
            imgsz=ov_cfg.get("imgsz", DEFAULT_IMGSZ),
            conf=conf,
            class_conf=class_conf,
            device=device,
            cpu_threads=threads,
        )

    classes = load_class_names(resolve_path(cfg.get("classes_file", "configs/classes.yaml"), root))
    allowed = coco_cfg.get("allowed_classes")

    if backend in ("ultralytics", "onnx"):
        return coco(allowed, onnx=backend == "onnx")
    if backend == "openvocab":
        return openvocab(classes)

    hybrid_cfg = cfg.get("hybrid", {})
    fast = coco(allowed, onnx=hybrid_cfg.get("coco_onnx", False))
    known = set(fast.names.values())
    coco_part = sorted(fast.allowed_names or known)
    rare = [c for c in classes if c not in known]
    logger.info("Hybrid: COCO model for %s; open-vocab for %s", ", ".join(coco_part), ", ".join(rare))
    return HybridDetector(
        fast,
        openvocab(rare),
        openvocab_every=hybrid_cfg.get("openvocab_every", 2),
        nms_iou=hybrid_cfg.get("nms_iou", 0.5),
        agnostic_iou=hybrid_cfg.get("agnostic_iou", 0.7),
        class_conf=class_conf,
        conf=conf,
    )
