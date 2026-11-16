"""Tests for scripts/compare_detectors.py: AP math, test-split extraction, tables.

No model is loaded; detectors are fakes.
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare_detectors as cd  # noqa: E402

from src.detection import Detection  # noqa: E402

CLASSES = ["person", "stairs", "pole"]


def d(name: str, conf: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(name, conf, *box)


# ---------------------------------------------------------------------- AP

def test_average_precision_edge_cases() -> None:
    assert cd.average_precision([True, True], [0.9, 0.8], n_gt=2) == pytest.approx(1.0)
    assert cd.average_precision([], [], n_gt=3) == 0.0
    assert math.isnan(cd.average_precision([False], [0.5], n_gt=0))
    # Half the objects found, at full precision: AP ~ 0.5 (51 of 101 recall points).
    assert cd.average_precision([True], [0.9], n_gt=2) == pytest.approx(51 / 101)


def test_average_precision_ranks_by_confidence() -> None:
    # A false positive ranked above the only hit halves precision at every recall.
    assert cd.average_precision([False, True], [0.9, 0.8], n_gt=1) == pytest.approx(0.5)
    assert cd.average_precision([True, False], [0.9, 0.8], n_gt=1) == pytest.approx(1.0)


def test_ap50_matches_each_ground_truth_once() -> None:
    gt = {"a": [("stairs", 0.1, 0.1, 0.5, 0.5)], "b": [("person", 0.5, 0.5, 0.9, 0.9)]}
    preds = {
        "a": [d("stairs", 0.9, (0.1, 0.1, 0.5, 0.5)), d("stairs", 0.8, (0.11, 0.1, 0.5, 0.5))],  # duplicate = FP
        "b": [d("person", 0.7, (0.0, 0.0, 0.2, 0.2))],  # wrong place = FP
    }
    ap = cd.ap50_per_class(preds, gt, CLASSES)
    assert ap["stairs"] == pytest.approx(1.0)  # TP ranked first; the FP comes after full recall
    assert ap["person"] == 0.0
    assert math.isnan(ap["pole"])


def test_ap50_needs_iou_half() -> None:
    gt = {"a": [("pole", 0.0, 0.0, 0.2, 1.0)]}
    shifted = {"a": [d("pole", 0.9, (0.1, 0.0, 0.3, 1.0))]}  # IoU 1/3
    assert cd.ap50_per_class(shifted, gt, ["pole"])["pole"] == 0.0


def test_yolo_label_boxes_reads_boxes_and_polygons(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    label.write_text("1 0.5 0.5 0.2 0.4\n2 0.1 0.1 0.3 0.1 0.3 0.6 0.1 0.6\n\n")
    boxes = cd.yolo_label_boxes(label)
    assert boxes[0] == (1, pytest.approx(0.4), pytest.approx(0.3), pytest.approx(0.6), pytest.approx(0.7))
    assert boxes[1] == (2, 0.1, 0.1, 0.3, 0.6)
    assert cd.yolo_label_boxes(tmp_path / "missing.txt") == []


# ---------------------------------------------------------------------- test split from the zip

def make_zip(tmp_path: Path, names: list[str] = CLASSES) -> Path:
    zip_path = tmp_path / "drive" / "merged.zip"
    zip_path.parent.mkdir()
    img = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", yaml.safe_dump({"path": "/content/datasets/merged", "names": dict(enumerate(names))}))
        for split in ("train", "val", "test"):
            zf.writestr(f"images/{split}/{split}_0.jpg", img)
            zf.writestr(f"labels/{split}/{split}_0.txt", "1 0.5 0.5 0.2 0.2\n")
        zf.writestr("merge_report.json", "{}")
    zip_path.with_suffix(".done").write_text(json.dumps({"zip_bytes": zip_path.stat().st_size}))
    return zip_path


def test_extract_test_split_only_takes_test_files(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path)
    dest = tmp_path / "merged_test"
    data_yaml = cd.extract_test_split(zip_path, dest, CLASSES)

    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert files == [".extracted", "data.yaml", "images/test/test_0.jpg", "labels/test/test_0.txt"]
    data = yaml.safe_load(data_yaml.read_text())
    assert data["path"] == str(dest.absolute())
    assert data["test"] == "images/test" and data["names"] == dict(enumerate(CLASSES))

    pairs = cd.load_split(data_yaml)
    assert pairs == [(dest / "images/test/test_0.jpg", dest / "labels/test/test_0.txt")]


def test_extract_test_split_reuses_earlier_extraction(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path)
    dest = tmp_path / "merged_test"
    cd.extract_test_split(zip_path, dest, CLASSES)
    (dest / "images/test/test_0.jpg").unlink()  # would come back if it extracted again
    cd.extract_test_split(zip_path, dest, CLASSES)
    assert not (dest / "images/test/test_0.jpg").exists()


def test_extract_test_split_needs_done_file(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path)
    zip_path.with_suffix(".done").unlink()
    with pytest.raises(RuntimeError, match="merged.done"):
        cd.extract_test_split(zip_path, tmp_path / "out", CLASSES)


def test_extract_test_split_rejects_other_class_list(tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path, names=["person", "pole", "stairs"])
    with pytest.raises(RuntimeError, match="differ"):
        cd.extract_test_split(zip_path, tmp_path / "out", CLASSES)


def test_class_list_matches_classes_yaml() -> None:
    names = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]
    assert len(names) == 20
    assert set(cd.RARE_CLASSES) <= set(names)


# ---------------------------------------------------------------------- frames and benchmark

class FakeDetector:
    device = "cpu"
    last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [d("stairs", 0.5, (0.1, 0.1, 0.2, 0.2)), d("person", 0.9, (0.5, 0.5, 0.6, 0.6))]


def test_iter_frames_reads_images_and_videos(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((32, 32, 3), np.uint8))
    writer = cv2.VideoWriter(str(tmp_path / "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    (tmp_path / "notes.txt").write_text("skip me")

    names = [name for name, _ in cd.iter_frames([tmp_path], max_frames=3)]
    assert names == ["a.jpg", "walk.mp4#0", "walk.mp4#1", "walk.mp4#2"]


def test_benchmark_counts_classes() -> None:
    result = cd.benchmark(FakeDetector(), [np.zeros((8, 8, 3), np.uint8)] * 3)
    assert result["frames"] == 3 and result["fps"] > 0
    assert result["counts"] == {"stairs": 3, "person": 3}


def test_evaluate_with_fake_detector(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), np.uint8))
    label = tmp_path / "a.txt"
    label.write_text("1 0.15 0.15 0.1 0.1\n")  # stairs at the box FakeDetector returns
    ap = cd.evaluate(FakeDetector(), [(img, label)], CLASSES)
    assert ap["stairs"] == pytest.approx(1.0)
    assert math.isnan(ap["person"])  # predicted but no ground truth


# ---------------------------------------------------------------------- tables

def test_map_table_marks_rare_and_unsupported_classes() -> None:
    maps = {
        "ultralytics": {"person": 0.8, "stairs": None, "pole": None},
        "hybrid": {"person": 0.8, "stairs": 0.4, "pole": math.nan},
    }
    table = cd.format_map_table(maps, CLASSES)
    lines = table.splitlines()
    assert lines[0].split() == ["class", "COCO-only", "mAP50", "hybrid", "mAP50"]
    stairs = next(line for line in lines if "stairs" in line)
    assert stairs.startswith("* stairs") and stairs.split()[-2:] == ["-", "0.400"]
    assert next(line for line in lines if "pole" in line).split()[-1] == "n/a"
    # COCO-only can't see stairs: counts as 0 in the rare mean.
    rare = next(line for line in lines if line.startswith("mean (rare"))
    assert rare.split()[-2:] == ["0.000", "0.400"]


def test_counts_table_lists_fps_and_classes() -> None:
    results = {"openvocab": {"frames": 2, "fps": 9.5, "mean_ms": 105, "max_ms": 130, "counts": {"pole": 3}}}
    table = cd.format_counts_table(results)
    assert "9.5" in table and "* pole" in table
