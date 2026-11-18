"""Tests for the fine-tuned backend and the per-class hybrid in src/detector.py (fake models)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import src.detector as detector_mod
from src.detection import Detection
from src.detector import Detector, HybridDetector, OpenVocabDetector, build_detector, resolve_choice

ROOT = Path(__file__).resolve().parent.parent
CLASSES = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]
CHOICE_FILE = ROOT / "models" / "budget_v1" / "per_class_choice.yaml"
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "fire hydrant",
              "stop sign", "bench", "dog", "chair", "potted plant"]
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=float)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeResult:
    def __init__(self, rows: Sequence[tuple]) -> None:
        self.boxes = type("Boxes", (), {"__len__": lambda self_: len(rows)})()
        self.boxes.cls = FakeTensor([r[0] for r in rows])
        self.boxes.conf = FakeTensor([r[1] for r in rows])
        self.boxes.xyxyn = FakeTensor([r[2:] for r in rows] or np.zeros((0, 4)))


class FakeYOLO:
    """best.pt -> our 20 classes; anything else -> COCO names. Rows: (class_id, conf, x1, y1, x2, y2)."""

    rows: dict[str, list[tuple]] = {}

    def __init__(self, path: str) -> None:
        self.path = path
        names = CLASSES if "best" in Path(path).name else COCO_NAMES
        self.names = dict(enumerate(names))
        self.calls = 0

    def predict(self, frame: np.ndarray, **kwargs: Any) -> list[FakeResult]:
        self.calls += 1
        return [FakeResult(FakeYOLO.rows.get(Path(self.path).name, []))]


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Default config pointing at a fake best.pt and a copy of the real per_class_choice.yaml."""
    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (FakeOV, FakeOV))
    FakeYOLO.rows = {}
    (tmp_path / "best.pt").write_text("weights")
    (tmp_path / "per_class_choice.yaml").write_text(CHOICE_FILE.read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["detector"]
    cfg["finetuned"] = {**cfg["finetuned"], "model_path": str(tmp_path / "best.pt"),
                        "per_class_choice": str(tmp_path / "per_class_choice.yaml")}
    cfg["coco"] = {**cfg["coco"], "model_path": str(tmp_path / "yolo11n.pt")}
    cfg["openvocab"] = {**cfg["openvocab"], "model_path": str(tmp_path / "yoloe-11s-seg.pt"),
                        "cache_dir": str(tmp_path / "cache")}
    return cfg


class FakeOV(FakeYOLO):
    """Stands in for YOLOE: prompted classes become its names."""

    def __init__(self, path: str = "yoloe.pt") -> None:
        super().__init__(path)
        cached = Path(path)
        self.names = dict(enumerate(cached.read_text().splitlines())) if cached.is_file() else {}

    def set_classes(self, classes: list[str]) -> None:
        self.names = dict(enumerate(classes))

    def save(self, path: str) -> None:
        Path(path).write_text("\n".join(self.names.values()))


# ---------------------------------------------------------------------- choice

def test_budget_v1_choice_file_is_committed_and_complete() -> None:
    doc = yaml.safe_load(CHOICE_FILE.read_text())
    assert doc["run"] == "budget_v1" and set(doc["scores"]) == set(CLASSES)


@pytest.mark.parametrize(("min_gain", "to_coco"), [
    (0.05, []),                # bench wins by 0.035, dog by 0.002: both under 0.05
    (0.03, ["bench"]),
    (0.0, ["dog", "bench"]),   # the file's own choice
])
def test_resolve_choice_with_budget_v1_scores(min_gain: float, to_coco: list[str]) -> None:
    route = resolve_choice(yaml.safe_load(CHOICE_FILE.read_text()), CLASSES, min_gain)
    assert [c for c in CLASSES if route[c] == "coco"] == to_coco
    assert all(route[c] == "finetuned" for c in CLASSES if c not in to_coco)


def test_resolve_choice_routes_openvocab_and_missing_scores() -> None:
    doc = {"scores": {"stairs": {"finetuned": 0.3, "zero_training": 0.5, "zero_training_model": "openvocab"},
                      "curb": {"finetuned": None, "zero_training": 0.9, "zero_training_model": "openvocab"}}}
    assert resolve_choice(doc, ["stairs", "curb", "pole"]) == {"stairs": "openvocab", "curb": "finetuned",
                                                              "pole": "finetuned"}
    with pytest.raises(ValueError, match="unknown"):
        resolve_choice({"scores": {"pole": {"finetuned": 0, "zero_training": 1, "zero_training_model": "x"}}}, ["pole"])


# ---------------------------------------------------------------------- finetuned backend

def test_finetuned_backend_uses_class_list_imgsz_and_thresholds(setup: dict) -> None:
    # stairs at 0.16 (critical, threshold 0.15) kept; bench at 0.20 (default 0.25) dropped.
    FakeYOLO.rows = {"best.pt": [(13, 0.16, 0.1, 0.5, 0.6, 1.0), (18, 0.20, 0.6, 0.6, 0.9, 0.9)]}
    det = build_detector(setup, "finetuned")
    assert isinstance(det, Detector) and det.imgsz == 512
    assert list(det.names.values()) == CLASSES
    assert [d.class_name for d in det.detect(FRAME)] == ["stairs"]
    assert det.predict_conf == pytest.approx(0.15)


def test_finetuned_rejects_model_with_other_class_count(setup: dict) -> None:
    (Path(setup["finetuned"]["model_path"]).parent / "wrong.pt").write_text("x")
    setup["finetuned"]["model_path"] = str(Path(setup["finetuned"]["model_path"]).parent / "wrong.pt")
    with pytest.raises(RuntimeError, match="13 classes"):
        build_detector(setup, "finetuned")


def test_finetuned_missing_file_is_an_error(setup: dict) -> None:
    setup["finetuned"]["model_path"] = "models/nope/best.pt"
    with pytest.raises(FileNotFoundError, match="notebook 05"):
        build_detector(setup, "finetuned")


# ---------------------------------------------------------------------- hybrid

def test_hybrid_default_is_the_finetuned_model_alone(setup: dict) -> None:
    det = build_detector(setup, "hybrid")
    assert isinstance(det, Detector) and det.model_path.name == "best.pt"  # no yolo11n, no YOLOE


def test_hybrid_with_min_gain_005_still_runs_one_model(setup: dict) -> None:
    setup["finetuned"]["only"] = False
    det = build_detector(setup, "hybrid")  # nothing beats the fine-tuned model by 0.05
    assert isinstance(det, Detector) and det.model_path.name == "best.pt"


def test_hybrid_routes_lost_classes_to_yolo11n_only(setup: dict) -> None:
    setup["finetuned"].update(only=False, min_gain=0.0)
    FakeYOLO.rows = {"best.pt": [(13, 0.9, 0.1, 0.5, 0.4, 1.0), (18, 0.9, 0.6, 0.6, 0.9, 0.9)],  # stairs, bench
                     "yolo11n.pt": [(9, 0.8, 0.6, 0.6, 0.9, 0.9), (2, 0.9, 0.0, 0.0, 0.2, 0.2)]}  # bench, car
    det = build_detector(setup, "hybrid")
    assert isinstance(det, HybridDetector) and det.slow is None  # YOLOE never loaded
    fine, coco = det.fast_members
    assert "bench" not in fine.allowed_names and "dog" not in fine.allowed_names  # type: ignore[attr-defined]
    assert coco.allowed_names == {"bench", "dog"}  # type: ignore[attr-defined]
    found = sorted(d.class_name for d in det.detect(FRAME))
    assert found == ["bench", "stairs"]  # bench from yolo11n only; yolo11n's car is not its class


def test_hybrid_loads_yoloe_only_for_classes_it_won(setup: dict, tmp_path: Path) -> None:
    doc = yaml.safe_load(CHOICE_FILE.read_text())
    doc["scores"]["curb"]["zero_training"] = 0.99
    (tmp_path / "per_class_choice.yaml").write_text(yaml.safe_dump(doc))
    setup["finetuned"].update(only=False)
    det = build_detector(setup, "hybrid")
    assert isinstance(det, HybridDetector) and isinstance(det.slow, OpenVocabDetector)
    assert det.slow.classes == ["curb"]
    assert len(det.fast_members) == 1  # no yolo11n
    assert det.class_conf["curb"] == setup["class_conf"]["curb"]  # YOLOE's threshold for its class
    assert det.class_conf["stairs"] == 0.15


def test_hybrid_without_finetuned_model_falls_back_and_logs(setup: dict, caplog: pytest.LogCaptureFixture) -> None:
    Path(setup["finetuned"]["model_path"]).unlink()
    with caplog.at_level(logging.WARNING):
        det = build_detector(setup, "hybrid")
    assert isinstance(det, HybridDetector) and isinstance(det.slow, OpenVocabDetector)
    assert "zero-training hybrid" in caplog.text


def test_hybrid_with_several_fast_models_and_no_slow() -> None:
    class Backend:
        device = "cpu"

        def __init__(self, dets: list[Detection], ms: float) -> None:
            self.dets, self.ms, self.last_inference_ms = dets, ms, 0.0

        def detect(self, frame: np.ndarray) -> list[Detection]:
            self.last_inference_ms = self.ms
            return self.dets

    a = Backend([Detection("stairs", 0.5, 0.1, 0.1, 0.3, 0.5)], 30)
    b = Backend([Detection("bench", 0.6, 0.6, 0.6, 0.9, 0.9)], 50)
    hybrid = HybridDetector([a, b])
    assert sorted(d.class_name for d in hybrid.detect(FRAME)) == ["bench", "stairs"]
    assert hybrid.last_inference_ms == pytest.approx(80, abs=5) and not hybrid.openvocab_ran
    with pytest.raises(ValueError):
        HybridDetector([])
