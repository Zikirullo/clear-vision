"""scripts/safety_report.py: matching, size groups, thresholds, PASS, tables, JSON, examples.

A tiny fake test split and fake predictions; no model is loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import safety_report as sr  # noqa: E402

from src.detection import Detection  # noqa: E402

TARGET = sr.SafetyTarget(classes=("curb", "stairs"), conf_levels=(0.15, 0.25))

# Boxes by share of the image area: small 0.5%, medium 4%, large 25%.
SMALL = (0.10, 0.10, 0.17071, 0.17071)
MEDIUM = (0.40, 0.40, 0.60, 0.60)
LARGE = (0.25, 0.25, 0.75, 0.75)


def d(name: str, conf: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(name, conf, *box)


def shifted(box: tuple[float, ...], dx: float) -> tuple[float, float, float, float]:
    return (box[0] + dx, box[1], box[2] + dx, box[3])


# ---------------------------------------------------------------------- size groups

def test_size_groups_and_boundaries() -> None:
    assert sr.size_group(SMALL, TARGET) == "small"
    assert sr.size_group(MEDIUM, TARGET) == "medium"
    assert sr.size_group(LARGE, TARGET) == "large"
    assert sr.size_group((0, 0, 0.1, 0.1), TARGET) == "medium"  # exactly 1%: not small
    assert sr.size_group((0, 0, 0.5, 0.1), TARGET) == "medium"  # exactly 5%: not large
    assert sr.size_group((0, 0, 0.5, 0.1001), TARGET) == "large"


# ---------------------------------------------------------------------- matching

def test_match_needs_iou_half() -> None:
    gt = [MEDIUM]
    assert sr.match([d("curb", 0.9, shifted(MEDIUM, 0.06))], gt, 0.5) == ([True], [True])  # IoU 0.54
    assert sr.match([d("curb", 0.9, shifted(MEDIUM, 0.08))], gt, 0.5) == ([False], [False])  # IoU 0.43


def test_match_one_prediction_per_box_highest_confidence_first() -> None:
    preds = [d("curb", 0.3, MEDIUM), d("curb", 0.9, MEDIUM)]
    found, hit = sr.match(preds, [MEDIUM], 0.5)
    assert found == [True] and hit == [False, True]  # the duplicate is a false positive


def test_match_prefers_the_box_it_overlaps_most() -> None:
    a, b = (0.0, 0.0, 0.4, 0.4), (0.3, 0.0, 0.7, 0.4)
    found, _ = sr.match([d("curb", 0.9, (0.31, 0.0, 0.71, 0.4)), d("curb", 0.8, a)], [a, b], 0.5)
    assert found == [True, True]


# ---------------------------------------------------------------------- evaluate

GT = {
    "a.jpg": [("curb", *SMALL), ("curb", *MEDIUM), ("stairs", *LARGE)],
    "b.jpg": [("curb", *LARGE), ("curb", *MEDIUM)],
    "c.jpg": [],
}
PREDS = {
    "a.jpg": [d("curb", 0.20, MEDIUM), d("stairs", 0.90, LARGE), d("curb", 0.50, (0.8, 0.8, 0.9, 0.9))],
    "b.jpg": [d("curb", 0.80, LARGE), d("stairs", 0.70, MEDIUM)],  # medium curb found as stairs: a miss
    "c.jpg": [d("stairs", 0.10, LARGE)],  # below both thresholds
}


def test_recall_by_size_and_precision_at_each_threshold() -> None:
    low = sr.evaluate(PREDS, GT, TARGET, 0.15)["classes"]
    assert low["curb"]["n"] == {"small": 1, "medium": 2, "large": 1}
    assert low["curb"]["recall"] == {"small": 0.0, "medium": 0.5, "large": 1.0, "medium+large": pytest.approx(2 / 3)}
    assert low["curb"]["precision"] == pytest.approx(2 / 3)  # 3 curb predictions, 2 found a box
    assert low["curb"]["pass"] is False
    assert low["stairs"]["recall"]["medium+large"] == 1.0 and low["stairs"]["pass"] is True
    assert low["stairs"]["precision"] == 0.5  # the stairs on the medium curb is wrong

    high = sr.evaluate(PREDS, GT, TARGET, 0.25)["classes"]
    assert high["curb"]["recall"]["medium"] == 0.0  # the 0.20 prediction is dropped
    assert high["curb"]["precision"] == 1 / 2


def test_misses_list_only_medium_and_large() -> None:
    misses = sr.evaluate(PREDS, GT, TARGET, 0.15)["misses"]["curb"]
    assert [(m["image"], m["size"]) for m in misses] == [("b.jpg", "medium")]


def test_no_boxes_means_no_verdict() -> None:
    target = sr.SafetyTarget(classes=("pothole",), conf_levels=(0.15,))
    r = sr.evaluate(PREDS, GT, target, 0.15)["classes"]["pothole"]
    assert r["recall"]["medium+large"] is None and r["pass"] is None and r["precision"] is None


def test_pass_at_exactly_the_target() -> None:
    gt = {f"{i}.jpg": [("curb", *MEDIUM)] for i in range(10)}
    preds = {f"{i}.jpg": [d("curb", 0.9, MEDIUM)] for i in range(9)}
    r = sr.evaluate(preds, gt, TARGET, 0.15)
    assert r["classes"]["curb"]["recall"]["medium+large"] == 0.9 and r["classes"]["curb"]["pass"] is True
    assert sr.failing_classes(r) == []


def test_small_misses_do_not_fail_a_class() -> None:
    gt = {"a.jpg": [("curb", *SMALL), ("curb", *LARGE)]}
    r = sr.evaluate({"a.jpg": [d("curb", 0.9, LARGE)]}, gt, TARGET, 0.15)["classes"]["curb"]
    assert r["recall"]["small"] == 0.0 and r["pass"] is True


# ---------------------------------------------------------------------- table, config, report

def test_table_has_every_class_and_verdict() -> None:
    table = sr.format_table(sr.evaluate(PREDS, GT, TARGET, 0.15), TARGET)
    curb = next(line for line in table.splitlines() if line.startswith("curb"))
    assert "1/2/1" in curb and "0.67" in curb and "FAIL" in curb
    assert "PASS" in next(line for line in table.splitlines() if line.startswith("stairs"))
    assert "1 of 2 classes pass" in table


def test_config_target_matches_the_proposal() -> None:
    t = sr.load_target()
    assert t.classes == ("curb", "stairs", "pothole", "pole", "manhole", "bollard", "barrier",
                         "traffic_cone", "scooter", "crosswalk", "person")
    assert t.conf_levels == (0.15, 0.25)
    assert (t.iou, t.small_max, t.large_min, t.min_recall) == (0.5, 0.01, 0.05, 0.9)


def test_model_label() -> None:
    assert sr.model_label(Path("/x/exports/budget_v1/best.pt")) == "budget_v1"
    assert sr.model_label(Path("models/my_model.pt")) == "my_model"


def test_report_json_name_and_content(tmp_path: Path) -> None:
    results = [sr.evaluate(PREDS, GT, TARGET, c) for c in TARGET.conf_levels]
    report = sr.build_report(Path("exports/budget_v1/best.pt"), 512, 3, TARGET, results, "2026-11-24")
    path = sr.save_report(report, tmp_path / "safety_report")
    assert path.name == "budget_v1_2026-11-24.json"
    saved = json.loads(path.read_text())
    assert set(saved["thresholds"]) == {"0.15", "0.25"} and saved["imgsz"] == 512
    assert saved["thresholds"]["0.15"]["classes"]["curb"]["pass"] is False


def test_ground_truth_from_label_files(tmp_path: Path) -> None:
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    img, lbl = tmp_path / "images" / "test" / "a.jpg", tmp_path / "labels" / "test" / "a.txt"
    lbl.write_text("12 0.5 0.5 0.2 0.2\n99 0.5 0.5 0.1 0.1\n")  # 12 = curb; 99 is not a class
    gt = sr.load_ground_truth([(img, lbl)], sr.load_class_names(ROOT / "configs" / "classes.yaml"), tmp_path)
    assert list(gt) == ["images/test/a.jpg"]
    (cls, *box), = gt["images/test/a.jpg"]
    assert cls == "curb" and box == pytest.approx(list(MEDIUM))


# ---------------------------------------------------------------------- examples

def test_pick_misses_prefers_different_images_and_is_repeatable() -> None:
    misses = [{"image": f"{i % 5}.jpg", "box": list(MEDIUM), "size": "medium"} for i in range(20)]
    first = sr.pick_misses(misses, 8)
    assert len(first) == 8 and len({m["image"] for m in first}) == 5
    assert first == sr.pick_misses(misses, 8)
    assert len(sr.pick_misses(misses[:3], 8)) == 3


def test_draw_miss_marks_the_box_in_red() -> None:
    image = np.zeros((200, 200, 3), np.uint8)
    out = sr.draw_miss(image, {"box": list(MEDIUM)}, "curb", [d("stairs", 0.7, LARGE)])
    assert out.shape == image.shape and image.max() == 0  # the input is not changed
    assert tuple(out[80, 100]) == (255, 0, 0)  # top edge of the missed box, RGB red
    assert tuple(out[50, 100]) == tuple(int(v) for v in cv2.cvtColor(np.uint8([[[0, 220, 255]]]), cv2.COLOR_BGR2RGB)[0, 0])
