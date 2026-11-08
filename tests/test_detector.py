"""Tests for src/detector.py.

The sample image is COCO val2017 #000000011197 (a street with cars, people
and a bicycle), stored in tests/assets/.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.detector import Detection, Detector

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = ROOT / "tests" / "assets" / "coco_street_000000011197.jpg"
CONFIG = ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def detector_cfg() -> dict:
    with CONFIG.open() as f:
        return yaml.safe_load(f)["detector"]


@pytest.fixture(scope="module")
def detector(detector_cfg: dict) -> Detector:
    cfg = dict(detector_cfg)
    cfg["model_path"] = str(ROOT / cfg["model_path"])
    return Detector.from_config(cfg)


@pytest.fixture(scope="module")
def street_image() -> np.ndarray:
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None, f"missing test image {SAMPLE_IMAGE}"
    return image


def test_detection_properties() -> None:
    d = Detection("person", 0.9, x1=0.2, y1=0.1, x2=0.4, y2=0.6)
    assert d.center_x == pytest.approx(0.3)
    assert d.width == pytest.approx(0.2)
    assert d.height == pytest.approx(0.5)
    assert d.area == pytest.approx(0.1)
    assert d.track_id is None


def test_finds_person_or_car(detector: Detector, street_image: np.ndarray) -> None:
    detections = detector.detect(street_image)
    names = {d.class_name for d in detections}
    assert names & {"person", "car"}, f"expected a person or car, got {names}"


def test_only_allowed_classes_and_normalized_boxes(
    detector: Detector, street_image: np.ndarray, detector_cfg: dict
) -> None:
    allowed = set(detector_cfg["allowed_classes"])
    for d in detector.detect(street_image):
        assert d.class_name in allowed
        assert 0.0 <= d.x1 < d.x2 <= 1.0
        assert 0.0 <= d.y1 < d.y2 <= 1.0
        assert d.confidence >= detector.conf


def test_inference_time_640px(
    detector: Detector, street_image: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    h, w = street_image.shape[:2]
    frame = cv2.resize(street_image, (640, round(640 * h / w)))
    times = []
    for _ in range(20):
        detector.detect(frame)
        times.append(detector.last_inference_ms)
    median = statistics.median(times)
    with capsys.disabled():
        print(
            f"\n[detector] device={detector.device} 640px frame: "
            f"median {median:.1f} ms, min {min(times):.1f} ms, max {max(times):.1f} ms"
        )
    assert median > 0
