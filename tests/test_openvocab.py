"""Tests for the open-vocab and hybrid backends in src/detector.py.

Every model here is a fake: no weights, text encoder or network are needed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import src.detector as detector_mod
from src.detection import Detection
from src.detector import (
    HybridDetector,
    OpenVocabDetector,
    build_detector,
    build_vocabulary,
    canonical_name,
    filter_by_conf,
    load_openvocab_model,
    load_prompts,
    merge_detections,
    nms,
    vocabulary_key,
)

ROOT = Path(__file__).resolve().parent.parent
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------- fakes

class FakeTensor:
    def __init__(self, values: Sequence[Any]) -> None:
        self.values = np.asarray(values, dtype=float)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self, rows: Sequence[tuple[int, float, float, float, float, float]]) -> None:
        self.cls = FakeTensor([r[0] for r in rows])
        self.conf = FakeTensor([r[1] for r in rows])
        self.xyxyn = FakeTensor([r[2:] for r in rows] or np.zeros((0, 4)))

    def __len__(self) -> int:
        return len(self.cls.values)


class FakeResult:
    def __init__(self, rows: Sequence[tuple[int, float, float, float, float, float]]) -> None:
        self.boxes = FakeBoxes(rows)
        self.masks = object()  # a seg model returns masks; the detector must ignore them


class FakeModel:
    """Stands in for YOLOE / YOLOWorld / YOLO. Rows are (class_idx, conf, x1, y1, x2, y2)."""

    instances: list[FakeModel] = []

    def __init__(self, path: str = "fake.pt", names: Sequence[str] | None = None) -> None:
        self.path = path
        self.names: dict[int, str] = dict(enumerate(names or []))
        self.rows: list[tuple[int, float, float, float, float, float]] = []
        self.set_classes_calls: list[list[str]] = []
        self.predict_calls: list[dict[str, Any]] = []
        FakeModel.instances.append(self)
        # A saved or exported fake "weights" file holds its class names, one per line.
        if Path(path).is_file():
            self.names = dict(enumerate(Path(path).read_text().splitlines()))

    def set_classes(self, classes: list[str]) -> None:
        self.set_classes_calls.append(list(classes))
        self.names = dict(enumerate(classes))

    def save(self, path: str) -> None:
        Path(path).write_text("\n".join(self.names.values()))

    def export(self, format: str, imgsz: int, **_: Any) -> str:  # noqa: A002
        # Like Ultralytics: written next to the weights path (relative to the cwd).
        out = Path(self.path).with_suffix(".onnx")
        out.write_text("\n".join(self.names.values()))
        return str(out)

    def predict(self, frame: np.ndarray, **kwargs: Any) -> list[FakeResult]:
        self.predict_calls.append(kwargs)
        return [FakeResult(self.rows)]


class FailingModel(FakeModel):
    def __init__(self, path: str = "fake.pt", names: Sequence[str] | None = None) -> None:
        raise RuntimeError("weights not available")


class FakeBackend:
    def __init__(self, detections: list[Detection], ms: float = 10.0) -> None:
        self.detections = detections
        self.device = "cpu"
        self.last_inference_ms = 0.0
        self.calls = 0
        self.ms = ms

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        self.last_inference_ms = self.ms
        return list(self.detections)


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    FakeModel.instances.clear()


@pytest.fixture
def fake_classes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """YOLOE and YOLOWorld are fakes; loading an ONNX file through YOLO() too."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (FakeModel, FakeModel))
    monkeypatch.setattr(detector_mod, "YOLO", FakeModel)


def det(name: str, conf: float = 0.5, x1: float = 0.1, y1: float = 0.1, x2: float = 0.3, y2: float = 0.5) -> Detection:
    return Detection(name, conf, x1, y1, x2, y2)


# ---------------------------------------------------------------------- vocabulary

def test_prompts_file_covers_every_class() -> None:
    classes = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]
    prompts = load_prompts(ROOT / "configs" / "prompts.yaml")
    assert set(prompts) == {canonical_name(c) for c in classes}
    assert prompts["traffic_cone"] == ["traffic cone"]
    assert prompts["manhole"][0] == "manhole cover"
    phrases, _ = build_vocabulary(list(prompts), prompts)
    assert len(phrases) == len(set(phrases))


def test_build_vocabulary_maps_every_phrase_back() -> None:
    phrases, index_to_class = build_vocabulary(
        ["stairs", "traffic_cone", "curb"], {"stairs": ["stairs", "staircase"], "traffic_cone": ["traffic cone"]}
    )
    assert phrases == ["stairs", "staircase", "traffic cone", "curb"]  # curb: no prompt -> its name
    assert index_to_class == ["stairs", "stairs", "traffic_cone", "curb"]


def test_build_vocabulary_rejects_duplicate_phrases() -> None:
    with pytest.raises(ValueError, match="post"):
        build_vocabulary(["pole", "bollard"], {"pole": ["post"], "bollard": ["post"]})


def test_vocabulary_key_changes_with_prompts_or_model() -> None:
    base = vocabulary_key("yoloe-11s-seg.pt", ["pole"], ["pole"])
    assert base == vocabulary_key("yoloe-11s-seg.pt", ["pole"], ["pole"])
    assert base != vocabulary_key("yoloe-11s-seg.pt", ["lamp post"], ["pole"])
    assert base != vocabulary_key("yolov8s-worldv2.pt", ["pole"], ["pole"])


# ---------------------------------------------------------------------- detector

def make_openvocab(rows: list[tuple[int, float, float, float, float, float]], **kwargs: Any) -> tuple[OpenVocabDetector, FakeModel]:
    prompts = {"pole": ["pole", "lamp post"], "traffic_cone": ["traffic cone"], "car": ["car"]}
    model = FakeModel(names=["pole", "lamp post", "traffic cone", "car"])
    model.rows = rows
    detector = OpenVocabDetector(["pole", "traffic_cone", "car"], prompts, model=model, warm_up=False, **kwargs)
    return detector, model


def test_openvocab_maps_phrases_to_our_classes() -> None:
    detector, model = make_openvocab([
        (2, 0.30, 0.1, 0.2, 0.3, 0.6),   # traffic cone
        (1, 0.25, 0.6, 0.0, 0.65, 0.9),  # lamp post -> pole
    ], class_conf={"pole": 0.15, "traffic_cone": 0.15})
    found = detector.detect(FRAME)
    assert [d.class_name for d in found] == ["traffic_cone", "pole"]
    assert found[0] == Detection("traffic_cone", pytest.approx(0.30), 0.1, 0.2, 0.3, 0.6)
    # One predict() threshold: the lowest of the per-class ones.
    assert model.predict_calls[0]["conf"] == pytest.approx(0.15)


def test_openvocab_applies_per_class_thresholds() -> None:
    detector, _ = make_openvocab([
        (0, 0.20, 0.1, 0.1, 0.2, 0.9),   # pole 0.20 >= 0.15: kept
        (3, 0.20, 0.5, 0.5, 0.9, 0.9),   # car 0.20 < 0.35: dropped
    ], class_conf={"pole": 0.15, "car": 0.35})
    assert [d.class_name for d in detector.detect(FRAME)] == ["pole"]


def test_openvocab_keeps_one_box_when_two_phrases_hit_one_object() -> None:
    detector, _ = make_openvocab([
        (0, 0.40, 0.10, 0.1, 0.20, 0.9),  # "pole"
        (1, 0.30, 0.11, 0.1, 0.21, 0.9),  # "lamp post", same object
    ], conf=0.1)
    found = detector.detect(FRAME)
    assert len(found) == 1 and found[0].confidence == pytest.approx(0.40)


def test_openvocab_rejects_model_with_wrong_class_count() -> None:
    with pytest.raises(RuntimeError, match="3 classes"):
        OpenVocabDetector(["pole"], {"pole": ["pole", "lamp post"]},
                          model=FakeModel(names=["a", "b", "c"]), warm_up=False)


# ---------------------------------------------------------------------- loading and caching

def test_first_load_prompts_once_and_caches(tmp_path: Path, fake_classes: None) -> None:
    phrases, index_to_class = ["pole", "traffic cone"], ["pole", "traffic_cone"]
    model, path = load_openvocab_model(phrases, index_to_class, tmp_path / "yoloe-11s-seg.pt", None, tmp_path / "cache")
    assert model.set_classes_calls == [phrases]
    assert path.parent == tmp_path / "cache" and path.suffix == ".pt" and path.exists()

    FakeModel.instances.clear()
    again, again_path = load_openvocab_model(phrases, index_to_class, tmp_path / "yoloe-11s-seg.pt", None, tmp_path / "cache")
    assert again_path == path
    assert again.set_classes_calls == []  # loaded from cache: no text encoder
    assert list(again.names.values()) == phrases


def test_changed_prompts_build_a_new_cache(tmp_path: Path, fake_classes: None) -> None:
    _, a = load_openvocab_model(["pole"], ["pole"], tmp_path / "yoloe-11s-seg.pt", None, tmp_path)
    _, b = load_openvocab_model(["lamp post"], ["pole"], tmp_path / "yoloe-11s-seg.pt", None, tmp_path)
    assert a != b


def test_falls_back_to_yolo_world_when_yoloe_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (FailingModel, FakeModel))
    model, path = load_openvocab_model(
        ["pole"], ["pole"], tmp_path / "yoloe-11s-seg.pt", tmp_path / "yolov8s-worldv2.pt", tmp_path / "cache"
    )
    assert path.name.startswith("yolov8s-worldv2-")
    assert model.set_classes_calls == [["pole"]]


def test_falls_back_to_yolo_world_when_ultralytics_has_no_yoloe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (None, FakeModel))
    _, path = load_openvocab_model(
        ["pole"], ["pole"], tmp_path / "yoloe-11s-seg.pt", tmp_path / "yolov8s-worldv2.pt", tmp_path / "cache"
    )
    assert path.name.startswith("yolov8s-worldv2-")


def test_raises_when_no_open_vocab_model_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (FailingModel, FailingModel))
    with pytest.raises(RuntimeError, match="No open-vocab model"):
        load_openvocab_model(["pole"], ["pole"], tmp_path / "yoloe-11s-seg.pt", tmp_path / "yolov8s-worldv2.pt", tmp_path)


def test_onnx_runtime_exports_after_set_classes(tmp_path: Path, fake_classes: None) -> None:
    model, path = load_openvocab_model(
        ["pole", "stairs"], ["pole", "stairs"], tmp_path / "yoloe-11s-seg.pt", None, tmp_path / "cache",
        runtime="onnx", imgsz=480,
    )
    assert path.suffix == ".onnx" and path.name.endswith("-480.onnx") and path.exists()
    assert list(model.names.values()) == ["pole", "stairs"]  # class names baked into the export

    FakeModel.instances.clear()
    _, again = load_openvocab_model(
        ["pole", "stairs"], ["pole", "stairs"], tmp_path / "yoloe-11s-seg.pt", None, tmp_path / "cache",
        runtime="onnx", imgsz=480,
    )
    assert again == path
    assert all(m.set_classes_calls == [] for m in FakeModel.instances)


# ---------------------------------------------------------------------- merging

def test_filter_by_conf_uses_class_threshold_or_default() -> None:
    kept = filter_by_conf([det("pole", 0.2), det("car", 0.3), det("chair", 0.45)], {"pole": 0.15, "car": 0.35}, 0.4)
    assert [d.class_name for d in kept] == ["pole", "chair"]


def test_nms_is_per_class_unless_agnostic() -> None:
    a, b = det("car", 0.9), det("truck", 0.6)  # same box
    assert nms([a, b], 0.5) == [a, b]
    assert nms([a, b], 0.5, class_agnostic=True) == [a]


def test_merge_removes_duplicates_and_compares_relative_scores() -> None:
    coco = [det("fire_hydrant", 0.50), det("car", 0.9, 0.6, 0.6, 0.9, 0.9)]
    openvocab = [det("bollard", 0.30, 0.1, 0.1, 0.3, 0.52), det("car", 0.8, 0.6, 0.6, 0.9, 0.9)]
    merged = merge_detections([coco, openvocab], 0.5, 0.7, {"fire_hydrant": 0.35, "bollard": 0.15}, 0.4)
    # bollard 0.30/0.15 = 2.0 beats fire_hydrant 0.50/0.35 = 1.4; the two cars become one.
    assert sorted(d.class_name for d in merged) == ["bollard", "car"]
    assert next(d for d in merged if d.class_name == "car").confidence == pytest.approx(0.9)


def test_merge_without_agnostic_keeps_different_classes() -> None:
    merged = merge_detections([[det("fire_hydrant", 0.5)], [det("bollard", 0.3)]], 0.5, None, {}, 0.4)
    assert len(merged) == 2


# ---------------------------------------------------------------------- hybrid

def test_hybrid_runs_openvocab_every_second_frame_and_reuses_it() -> None:
    fast = FakeBackend([det("person", 0.8, 0.6, 0.1, 0.8, 0.9)], ms=40)
    slow = FakeBackend([det("stairs", 0.3, 0.1, 0.5, 0.5, 1.0)], ms=80)
    hybrid = HybridDetector(fast, slow, openvocab_every=2)

    names_per_frame, ms_per_frame = [], []
    for _ in range(4):
        names_per_frame.append(sorted(d.class_name for d in hybrid.detect(FRAME)))
        ms_per_frame.append(hybrid.last_inference_ms)

    assert fast.calls == 4 and slow.calls == 2
    assert names_per_frame == [["person", "stairs"]] * 4  # stairs reused on frames 2 and 4
    assert ms_per_frame[0] == pytest.approx(120, abs=5) and ms_per_frame[1] == pytest.approx(40, abs=5)


def test_hybrid_reset_forgets_old_openvocab_boxes() -> None:
    slow = FakeBackend([det("stairs")])
    hybrid = HybridDetector(FakeBackend([]), slow, openvocab_every=5)
    hybrid.detect(FRAME)
    slow.detections = []
    hybrid.reset()
    assert hybrid.detect(FRAME) == []
    assert slow.calls == 2


def test_hybrid_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        HybridDetector(FakeBackend([]), FakeBackend([]), openvocab_every=0)


# ---------------------------------------------------------------------- factory

COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "fire hydrant",
              "stop sign", "bench", "dog", "chair", "potted plant"]


class FakeCocoModel(FakeModel):
    def __init__(self, path: str = "yolo11n.pt", names: Sequence[str] | None = None) -> None:
        super().__init__(path, names or COCO_NAMES)


def test_build_hybrid_splits_classes_between_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_mod, "YOLO", FakeCocoModel)
    monkeypatch.setattr(detector_mod, "_model_classes", lambda: (FakeModel, FakeModel))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["detector"]
    cfg["openvocab"] = {**cfg["openvocab"], "cache_dir": str(tmp_path / "cache"),
                        "model_path": str(tmp_path / "yoloe-11s-seg.pt")}
    cfg["coco"] = {**cfg["coco"], "model_path": str(tmp_path / "yolo11n.pt")}

    hybrid = build_detector(cfg, "hybrid")
    assert isinstance(hybrid, HybridDetector)
    assert hybrid.openvocab_every == cfg["hybrid"]["openvocab_every"]
    assert isinstance(hybrid.slow, OpenVocabDetector)
    # The open-vocab model only gets the classes COCO doesn't know.
    assert set(hybrid.slow.classes) == {"scooter", "pole", "bollard", "traffic_cone", "barrier", "curb",
                                        "stairs", "pothole", "manhole", "crosswalk"}
    prompted = [m for m in FakeModel.instances if m.set_classes_calls]
    assert len(prompted) == 1 and "traffic cone" in prompted[0].set_classes_calls[0]
    # COCO names come out in our spelling.
    assert "fire_hydrant" in hybrid.fast.allowed_names  # type: ignore[attr-defined]


def test_build_detector_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="hybrid"):
        build_detector({}, "tensorrt")


# ---------------------------------------------------------------------- main.py

def test_main_parses_backend_flag() -> None:
    from src.main import parse_args

    assert parse_args(["--backend", "hybrid"]).backend == "hybrid"
    with pytest.raises(SystemExit):
        parse_args(["--backend", "nope"])


def test_app_falls_back_to_coco_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.main as main_mod

    coco = FakeBackend([])

    def fake_build(cfg: dict, backend: str | None = None, **_: Any) -> FakeBackend:
        if backend == "hybrid":
            raise RuntimeError("no internet for the text encoder")
        return coco

    monkeypatch.setattr(main_mod, "build_detector", fake_build)
    monkeypatch.setattr(main_mod, "LOG_DIR", tmp_path)
    args = argparse.Namespace(source=None, config=None, no_window=True, mute=True, backend="hybrid")
    app = main_mod.App(args, {"detector": {}})
    try:
        assert app._load_detector() is coco
        assert app.last_msg is not None and app.last_msg.priority == "urgent"
        assert "Hazard detector failed" in app.last_msg.text
    finally:
        app.session.close()
