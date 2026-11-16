"""Tests for scripts/train_utils.py (no training, no real model)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

import scripts.train_utils as tu
from scripts.compare_detectors import load_split
from scripts.merge_datasets import merge
from src.colab_data import prepare_dataset
from src.detection import Detection

ASSETS = Path(__file__).parent / "assets" / "merge"
CLASSES = ["person", "car", "bus", "pole", "stairs"]


# ---------------------------------------------------------------------- presets

def test_presets_match_the_plan() -> None:
    b, n = tu.PRESETS["BUDGET"], tu.PRESETS["NORMAL"]
    assert (b["MODEL"], b["IMGSZ"], b["EPOCHS"], b["SUBSET_FRACTION"], b["FREEZE"], b["BATCH"], b["PATIENCE"]) == \
        ("yolo11n.pt", 512, 30, 0.3, 10, -1, 10)
    assert (n["IMGSZ"], n["EPOCHS"], n["SUBSET_FRACTION"], n["FREEZE"], n["PATIENCE"]) == (640, 50, 0.5, None, 15)
    assert tu.AUGMENT["flipud"] == 0.0 and tu.AUGMENT["close_mosaic"] == 5 and tu.AUGMENT["mosaic"] == 1.0


# ---------------------------------------------------------------------- subset

def subset_input() -> tuple[list[str], dict[str, set[int]]]:
    """20 person-only, 10 car, 3 pole (one listed 3x as in a balanced list), 2 stairs, 5 background."""
    classes: dict[str, set[int]] = {}
    lines: list[str] = []

    def add(name: str, ids: set[int], times: int = 1) -> None:
        classes[name] = ids
        lines.extend([f"./images/train/{name}"] * times)

    for i in range(20):
        add(f"p{i}.jpg", {0})
    for i in range(10):
        add(f"c{i}.jpg", {1})
    add("pole0.jpg", {3}, times=3)
    add("pole1.jpg", {3, 0})
    add("pole2.jpg", {3})
    add("st0.jpg", {4})
    add("st1.jpg", {4, 1})
    for i in range(5):
        add(f"bg{i}.jpg", set())
    return lines, classes


def test_subset_keeps_every_focus_image_with_repeats() -> None:
    lines, classes = subset_input()
    out, summary = tu.build_subset(lines, classes, CLASSES, fraction=0.2, focus=["pole", "stairs"], person_cap=100)
    names = [ln.rsplit("/", 1)[1] for ln in out]
    assert names.count("pole0.jpg") == 3  # repeats survive
    assert {"pole1.jpg", "pole2.jpg", "st0.jpg", "st1.jpg"} <= set(names)
    assert summary["focus_images"] == 5
    assert summary["other_images"] == 35 and summary["other_kept"] == 7  # round(35 * 0.2)
    assert summary["subset_images"] == 12 and summary["subset_lines"] == len(out) == 14


def test_subset_caps_person_only_images() -> None:
    lines, classes = subset_input()
    out, summary = tu.build_subset(lines, classes, CLASSES, fraction=1.0, focus=["pole"], person_cap=4)
    names = {ln.rsplit("/", 1)[1] for ln in out}
    assert sum(1 for n in names if n.startswith("p") and not n.startswith("pole")) == 4
    assert summary["person_only_dropped"] == 16
    assert "pole1.jpg" in names  # pole + person is a focus image, never capped


def test_subset_is_deterministic_and_validates_fraction() -> None:
    lines, classes = subset_input()
    a, _ = tu.build_subset(lines, classes, CLASSES, 0.3, seed=7)
    b, _ = tu.build_subset(lines, classes, CLASSES, 0.3, seed=7)
    assert a == b
    with pytest.raises(ValueError):
        tu.build_subset(lines, classes, CLASSES, 0.0)


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    zip_path = tmp_path / "drive" / "merged.zip"
    merge(raw=ASSETS / "raw", zip_path=zip_path, work=tmp_path / "work",
          classes_path=ASSETS / "classes.yaml", mappings_dir=ASSETS / "mappings", sync=lambda: None)
    return prepare_dataset(zip_path, tmp_path / "content" / "merged", check=False).parent


def test_subset_files_on_a_real_dataset_pass_ultralytics(dataset: Path) -> None:
    from ultralytics.data.utils import check_det_dataset

    names = ["person", "car", "traffic_cone"]
    lines = tu.train_lines(dataset, balanced=False)
    assert len(lines) == 5 and all(ln.startswith("./images/train/") for ln in lines)
    image_classes = tu.train_image_classes(dataset, len(names))
    assert image_classes["src_a_a1.jpg"] == {0, 1} and image_classes["src_a_a3.jpg"] == set()
    subset, summary = tu.build_subset(lines, image_classes, names, 0.5, focus=["traffic_cone"])
    assert "./images/train/src_b_img2_jpg.rf.ccc.jpg" in subset  # the only train cone image
    data_yaml = tu.write_subset(dataset, dataset / "data.yaml", subset)
    data = check_det_dataset(str(data_yaml))
    assert data["train"] == str(dataset / "train_subset.txt")
    assert data["val"] == str(dataset / "images" / "val")  # val/test unchanged


def test_train_lines_reads_balanced_list(dataset: Path) -> None:
    (dataset / "train_balanced.txt").write_text("./images/train/a.jpg\n./images/train/a.jpg\n\n")
    assert tu.train_lines(dataset, balanced=True) == ["./images/train/a.jpg"] * 2


# ---------------------------------------------------------------------- training helpers

def test_epoch_timer_prints_once_with_estimate(capsys: pytest.CaptureFixture[str]) -> None:
    timer = tu.EpochTimer()
    timer.start -= 120  # the first epoch "took" 2 minutes
    timer(SimpleNamespace(epoch=0, epochs=30))
    timer(SimpleNamespace(epoch=1, epochs=30))
    out = capsys.readouterr().out
    assert out.count("took") == 1
    assert "Epoch 1/30 took 2.0 min" in out and "remaining 29 epochs: 1.0 h" in out


def test_epoch_timer_after_resume_counts_from_current_epoch() -> None:
    timer = tu.EpochTimer()
    timer.start -= 60
    timer(SimpleNamespace(epoch=19, epochs=30))
    assert "Epoch 20/30" in timer.messages[0] and "remaining 10 epochs: 0.2 h" in timer.messages[0]


def test_last_checkpoint(tmp_path: Path) -> None:
    assert tu.last_checkpoint(tmp_path, "budget_v1") is None
    (tmp_path / "budget_v1" / "weights").mkdir(parents=True)
    (tmp_path / "budget_v1" / "weights" / "last.pt").write_text("x")
    assert tu.last_checkpoint(tmp_path, "budget_v1") == tmp_path / "budget_v1" / "weights" / "last.pt"


def test_check_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**8, 10**8))
    assert tu.check_space(tmp_path / "not" / "yet", 50 * 10**6) == 10**8
    with pytest.raises(RuntimeError, match="Trash"):
        tu.check_space(tmp_path, 200 * 10**6)


# ---------------------------------------------------------------------- evaluation

def d(cls: str, conf: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(cls, conf, *box)


def test_find_misses_at_deployment_threshold() -> None:
    gt = {"a.jpg": [("curb", 0.0, 0.8, 1.0, 1.0), ("pole", 0.4, 0.0, 0.5, 1.0), ("car", 0.6, 0.6, 0.9, 0.9)],
          "b.jpg": [("stairs", 0.1, 0.5, 0.9, 1.0)]}
    preds = {"a.jpg": [d("curb", 0.10, (0.0, 0.8, 1.0, 1.0)),      # right place, too unsure -> miss
                       d("pole", 0.60, (0.41, 0.0, 0.51, 1.0))],   # found
             "b.jpg": [d("curb", 0.70, (0.1, 0.5, 0.9, 1.0))]}     # stairs seen as curb -> miss
    misses = tu.find_misses(preds, gt, ["curb", "pole", "stairs"])
    assert [(m.image, m.cls) for m in misses] == [("a.jpg", "curb"), ("b.jpg", "stairs")]
    assert misses[1].confused_as == "curb" and misses[0].confused_as is None
    assert tu.group_misses(misses) == {"curb": 1, "stairs": 1}


def test_miss_grid(tmp_path: Path) -> None:
    import cv2

    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((60, 80, 3), np.uint8))
    misses = [tu.Miss(str(img), "curb", (0.0, 0.8, 1.0, 1.0), 0.0, None)] * 5
    grid = tu.miss_grid(misses, {str(img): [d("car", 0.9, (0.1, 0.1, 0.2, 0.2))]})
    assert grid.shape == (600, 1600, 3)  # 5 misses -> 2 rows of 4
    assert tu.miss_grid([], {}) is None


class FakeDetector:
    device = "cpu"
    last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [d("traffic_cone", 0.9, (0.45, 0.45, 0.55, 0.55))]


def test_predict_split_on_real_test_split(dataset: Path) -> None:
    pairs = load_split(dataset / "data.yaml", "test")
    preds, gts = tu.predict_split(FakeDetector(), pairs, ["person", "car", "traffic_cone"])
    (key,) = gts
    assert gts[key][0][0] == "traffic_cone" and preds[key][0].class_name == "traffic_cone"


# ---------------------------------------------------------------------- comparison and choice

def test_comparison_and_choice() -> None:
    zero = {"curb": 0.001, "traffic_cone": 0.77, "person": 0.39, "pole": None, "stairs": float("nan")}
    fine = {"curb": 0.35, "traffic_cone": 0.70, "person": 0.39, "pole": 0.2, "stairs": None}
    classes = ["curb", "traffic_cone", "person", "pole", "stairs"]
    rows = tu.comparison_rows(classes, zero, fine, recall={"curb": 0.5})
    assert [r["better?"] for r in rows] == ["yes", "no", "same", "n/a", "n/a"]
    assert rows[0]["recall"] == 0.5 and rows[0]["critical"] and not rows[1]["critical"]
    table = tu.format_comparison(rows)
    assert table.splitlines()[0].startswith("class") and "! curb" in table and "  traffic_cone" in table

    result = tu.per_class_choice(classes, zero, fine, coco_classes=["person"])
    assert result["choice"] == {"curb": "finetuned", "traffic_cone": "openvocab", "person": "finetuned",
                                "pole": "finetuned", "stairs": "finetuned"}
    assert result["scores"]["person"]["zero_training_model"] == "coco"
    assert result["scores"]["stairs"] == {"finetuned": None, "zero_training": None, "zero_training_model": "openvocab"}
    yaml.safe_dump(result)  # must be plain YAML


def test_load_zero_training(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"map50": {"hybrid": {"curb": 0.001, "car": None}, "openvocab": {}}}))
    assert tu.load_zero_training(path) == {"curb": 0.001, "car": None}


def test_per_class_from_ultralytics() -> None:
    box = SimpleNamespace(ap_class_index=np.array([0, 3]),
                          class_result=lambda i: [(0.8, 0.7, 0.75, 0.5), (0.4, 0.3, 0.2, 0.1)][i])
    metrics: Any = SimpleNamespace(names={0: "person", 1: "car", 2: "bus", 3: "pole"}, box=box)
    out = tu.per_class_from_ultralytics(metrics, CLASSES)
    assert out == {"person": {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.5},
                   "pole": {"precision": 0.4, "recall": 0.3, "map50": 0.2, "map50_95": 0.1}}
