"""Object detection with Ultralytics YOLO.

Wraps a YOLO model and returns plain `Detection` objects with coordinates
normalized to 0-1, so downstream code (tracker, navigator) never depends on
frame resolution or on Ultralytics types.
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from src.detection import Detection

__all__ = ["Detection", "Detector", "select_device"]

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/yolo11n.pt"
DEFAULT_CONF = 0.4
DEFAULT_IMGSZ = 640


MIN_MACOS_FOR_MPS = 15


def _macos_major() -> int:
    version = platform.mac_ver()[0]
    return int(version.split(".")[0]) if version else 0


def select_device(preference: str = "auto") -> str:
    """Return "mps" on Apple Silicon when usable, else "cpu".

    Recent PyTorch MPS kernels call macOS 15 APIs; on macOS 14 they abort the
    whole process with a native exception that Python cannot catch, so "auto"
    only picks MPS on macOS 15+.
    """
    if preference != "auto":
        return preference
    if torch.backends.mps.is_available() and _macos_major() >= MIN_MACOS_FOR_MPS:
        return "mps"
    return "cpu"


class Detector:
    """YOLO detector filtered to the classes that matter for navigation.

    Args:
        model_path: Path to YOLO weights. Official model names (e.g. yolo11n.pt)
            are downloaded into that path's folder if missing.
        conf: Minimum confidence to keep a detection.
        allowed_classes: Class names to keep; None keeps every class.
        imgsz: Inference size in pixels.
        device: "auto", "mps" or "cpu".
        cpu_threads: Torch CPU threads; None keeps the torch default.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        conf: float = DEFAULT_CONF,
        allowed_classes: Iterable[str] | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "auto",
        cpu_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = select_device(device)
        self.last_inference_ms: float = 0.0
        if cpu_threads:
            torch.set_num_threads(cpu_threads)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(self.model_path))
        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids = self._resolve_class_ids(allowed_classes)
        self.allowed_names = (
            None if self.class_ids is None else {self.names[i] for i in self.class_ids}
        )

        logger.info(
            "Detector ready: model=%s device=%s threads=%d conf=%.2f imgsz=%d classes=%s",
            self.model_path, self.device, torch.get_num_threads(), self.conf, self.imgsz,
            "all" if self.class_ids is None else len(self.class_ids),
        )
        self._warm_up()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Detector:
        """Build from the `detector:` section of configs/default.yaml."""
        return cls(
            model_path=cfg.get("model_path", DEFAULT_MODEL_PATH),
            conf=cfg.get("conf", DEFAULT_CONF),
            allowed_classes=cfg.get("allowed_classes"),
            imgsz=cfg.get("imgsz", DEFAULT_IMGSZ),
            device=cfg.get("device", "auto"),
            cpu_threads=cfg.get("cpu_threads"),
        )

    def _resolve_class_ids(self, allowed: Iterable[str] | None) -> list[int] | None:
        if allowed is None:
            return None
        by_name = {name: idx for idx, name in self.names.items()}
        ids: list[int] = []
        for name in allowed:
            if name in by_name:
                ids.append(by_name[name])
            else:
                logger.warning("Unknown class %r is not in the model; ignoring it", name)
        return ids

    def _warm_up(self) -> None:
        # The first passes are slow (weight transfer, MPS kernel compilation);
        # do them now so the live loop starts at full speed.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.detect(dummy)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame and return allowed-class detections."""
        start = time.perf_counter()
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )[0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxyn = boxes.xyxyn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        detections = [
            Detection(
                class_name=self.names[int(c)],
                confidence=float(p),
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
            )
            for b, p, c in zip(xyxyn, confs, cls_ids)
        ]
        # Some backends (e.g. exported ONNX models) ignore `classes=`, so filter
        # again here rather than trusting predict().
        if self.allowed_names is not None:
            detections = [d for d in detections if d.class_name in self.allowed_names]
        return detections
