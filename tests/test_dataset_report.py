"""Tests for scripts/dataset_report.py and src/colab_data.py apply_balanced().

Uses the merged.zip built by the real merge from tests/assets/merge (classes person,
car, traffic_cone), optionally pseudo-labeled with a fake model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import scripts.dataset_report as dr
from scripts.merge_datasets import merge
from scripts.pseudo_label import pseudo_label
from src.colab_data import apply_balanced, prepare_dataset

ASSETS = Path(__file__).parent / "assets" / "merge"
CLASSES = ["person", "car", "traffic_cone"]


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=float)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeResult:
    def __init__(self, rows: list) -> None:
        self.boxes = type("B", (), {})()
        self.boxes.cls = FakeTensor([r[0] for r in rows])
        self.boxes.conf = FakeTensor([r[1] for r in rows])
        self.boxes.xywhn = FakeTensor([r[2:] for r in rows] or np.zeros((0, 4)))
        self.boxes.__class__.__len__ = lambda self_: len(rows)


class FakeModel:
    names = {0: "person", 2: "car"}
    ckpt_path = "yolo11x.pt"

    def predict(self, sources: list[str], **_: Any) -> list[FakeResult]:
        rows = {"src_b_img3_jpg.rf.ddd": [(0, 0.9, 0.5, 0.5, 0.3, 0.6)],   # person in a train image
                "src_b_img4_jpg.rf.eee": [(2, 0.8, 0.2, 0.5, 0.1, 0.1)]}   # car in the test image
        return [FakeResult(rows.get(Path(s).stem, [])) for s in sources]


@pytest.fixture(autouse=True)
def no_font_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)


@pytest.fixture
def drive_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "drive" / "merged.zip"
    merge(raw=ASSETS / "raw", zip_path=zip_path, work=tmp_path / "work",
          classes_path=ASSETS / "classes.yaml", mappings_dir=ASSETS / "mappings", sync=lambda: None)
    return zip_path


@pytest.fixture
def dataset(drive_zip: Path, tmp_path: Path) -> Path:
    return prepare_dataset(drive_zip, tmp_path / "content" / "merged", check=False).parent


def report_of(zip_path: Path) -> dict:
    return json.loads(zip_path.with_name("merge_report.json").read_text())


def scan_of(dataset: Path, zip_path: Path, pseudo: bool = False) -> dr.Scan:
    csv_path = zip_path.with_name("pseudo_labels.csv") if pseudo else None
    return dr.scan_dataset(dataset, CLASSES, list(report_of(zip_path)["sources"]), csv_path)


# ---------------------------------------------------------------------- helpers

def test_size_bins_use_training_size() -> None:
    # 1280x720 image shrunk to 640x360: a 0.05 x 0.05 box is 32 x 18 px -> small.
    assert dr.size_bin(0.05, 0.05, 1280, 720) == "small"
    assert dr.size_bin(0.15, 0.15, 1280, 720) == "medium"  # 96 x 54 px
    assert dr.size_bin(0.2, 0.2, 1280, 720) == "large"  # 128 x 72 px = 96^2 exactly
    assert dr.size_bin(0.5, 0.5, 640, 640) == "large"


def test_source_of_longest_prefix() -> None:
    assert dr.source_of("rf_scooter_bollard_x", ["rf_scooter", "rf_scooter_bollard"]) == "rf_scooter_bollard"
    assert dr.source_of("x_1", ["rf_cone"]) == "unknown"


# ---------------------------------------------------------------------- scan and counts

def test_scan_counts_images_boxes_and_sources(dataset: Path, drive_zip: Path) -> None:
    scan = scan_of(dataset, drive_zip)
    report = report_of(drive_zip)
    assert dr.images_per_split(scan) == report["images"]
    per = dr.boxes_per_class(scan)
    for split in dr.SPLITS:
        for c in CLASSES:
            assert per[split][c] == {"original": report["boxes"][split][c], "pseudo": 0}
    assert dr.boxes_per_source(scan)["traffic_cone"] == {"src_b": 4}
    assert set(dr.boxes_per_source(scan)["person"]) == {"src_a"}
    sizes = dr.size_distribution(scan)
    assert sum(sum(v.values()) for v in sizes.values()) == len(scan.boxes)


def test_pseudo_labels_are_counted_separately(drive_zip: Path, tmp_path: Path) -> None:
    pseudo_label(drive_zip, tmp_path / "content" / "merged", model=FakeModel(), pseudo_classes=("person", "car"),
                 coco_sources=("src_a",), sync=lambda: None, check_dataset=False)
    dataset = tmp_path / "content" / "merged"
    scan = scan_of(dataset, drive_zip, pseudo=True)
    per = dr.boxes_per_class(scan)
    assert per["train"]["person"]["pseudo"] == 1
    assert per["test"]["car"] == {"original": 0, "pseudo": 1}
    assert per["test"]["traffic_cone"] == {"original": 1, "pseudo": 0}  # the original line before it
    assert not scan.pseudo_mismatch
    assert dr.boxes_per_source(scan)["person"] == {"src_a": 1, "src_b": 1}


def test_class_table_marks_classes_under_300_train_boxes(dataset: Path, drive_zip: Path) -> None:
    scan = scan_of(dataset, drive_zip)
    rows = dr.class_table(scan)
    assert [r["class"] for r in rows] == CLASSES
    assert {r["status"] for r in rows} == {"needs more data"}  # the fixture is tiny
    assert {r["status"] for r in dr.class_table(scan, min_train_boxes=1)} == {"ok"}
    table = dr.format_rows(rows)
    assert table.splitlines()[0].split()[:3] == ["class", "train", "boxes"]


# ---------------------------------------------------------------------- sanity checks

def test_clean_dataset_passes(dataset: Path, drive_zip: Path) -> None:
    checks = dr.sanity_checks(dataset, scan_of(dataset, drive_zip), report_of(drive_zip))
    assert [c.status for c in checks] == ["PASS", "PASS", "PASS", "PASS"]


def test_image_in_two_splits_fails(dataset: Path, drive_zip: Path) -> None:
    # Same content under another name in val.
    src = next((dataset / "images" / "train").glob("*.jpg"))
    shutil.copy(src, dataset / "images" / "val" / "copy_of_train.jpg")
    checks = dr.sanity_checks(dataset, scan_of(dataset, drive_zip), report_of(drive_zip))
    leak = checks[0]
    assert leak.status == "FAIL" and "1 identical files" in leak.detail
    assert checks[3].status == "FAIL"  # and the counts no longer match the report


def test_same_name_in_two_splits_fails_without_hashing(dataset: Path, drive_zip: Path) -> None:
    src = next((dataset / "images" / "train").glob("*.jpg"))
    shutil.copy(src, dataset / "images" / "test" / src.name)
    checks = dr.sanity_checks(dataset, scan_of(dataset, drive_zip), report_of(drive_zip), hash_images=False)
    assert checks[0].status == "FAIL" and checks[0].detail.startswith("1 names")


def test_class_id_out_of_range_fails(dataset: Path, drive_zip: Path) -> None:
    label = dataset / "labels" / "train" / "src_a_a2.txt"
    label.write_text(label.read_text() + "19 0.5 0.5 0.1 0.1\nnot a box\n")
    checks = dr.sanity_checks(dataset, scan_of(dataset, drive_zip), report_of(drive_zip))
    assert checks[1].status == "FAIL"
    assert "1 out-of-range IDs, 1 malformed lines" in checks[1].detail and "class 19" in checks[1].detail


def test_empty_val_label_warns(dataset: Path, drive_zip: Path) -> None:
    (dataset / "labels" / "val" / "src_b_img1_jpg.rf.aaa.txt").write_text("")
    checks = dr.sanity_checks(dataset, scan_of(dataset, drive_zip), report_of(drive_zip))
    assert checks[2].status == "WARN" and "val 1, test 0" in checks[2].detail


# ---------------------------------------------------------------------- balancing

def synthetic_scan() -> dr.Scan:
    """10 person-only, 2 pole, 1 bus+person, 6 car, 1 background train images."""
    classes = ["person", "car", "bus", "pole"]
    images: list[str] = []
    image_classes: dict[tuple[str, str], set[int]] = {}
    boxes: list[dr.BoxRow] = []

    def add(name: str, *cls: int) -> None:
        images.append(name)
        image_classes[("train", name)] = set(cls)
        boxes.extend(dr.BoxRow("train", name, "src", c, 0.1, 0.1, False, "small") for c in cls)

    for i in range(10):
        add(f"p{i}.jpg", 0)
    add("pole0.jpg", 3)
    add("pole1.jpg", 3, 1)
    add("bus0.jpg", 2, 0)
    for i in range(6):
        add(f"car{i}.jpg", 1)
    add("bg.jpg")
    return dr.Scan(classes=classes, images={"train": images, "val": [], "test": []},
                   image_classes=image_classes, boxes=boxes, empty_labels={s: [] for s in dr.SPLITS})


def test_repeat_factors_clamped_to_max() -> None:
    factors, target = dr.repeat_factors(synthetic_scan(), focus=["pole", "bus"], target_images=8)
    assert target == 8
    assert factors == {"pole": 4, "bus": 4}  # ceil(8/2)=4, ceil(8/1)=8 -> clamped to 4
    factors, _ = dr.repeat_factors(synthetic_scan(), focus=["pole"], target_images=3)
    assert factors == {"pole": 2}
    with pytest.raises(ValueError):
        dr.repeat_factors(synthetic_scan(), focus=["curb"])


def test_balanced_train_repeats_rare_and_caps_person_only() -> None:
    lines, summary = dr.balanced_train(synthetic_scan(), focus=["pole", "bus"], person_cap=5, target_images=8)
    assert all(ln.startswith("./images/train/") for ln in lines)
    names = [ln.rsplit("/", 1)[1] for ln in lines]
    assert names.count("pole0.jpg") == 4 and names.count("pole1.jpg") == 4 and names.count("bus0.jpg") == 4
    assert sum(1 for n in set(names) if n.startswith("p") and not n.startswith("pole")) == 5  # capped
    assert names.count("car0.jpg") == 1 and names.count("bg.jpg") == 1
    assert summary["person_only_images"] == 10 and summary["person_only_dropped"] == 5
    assert len(lines) == 5 + 4 + 4 + 4 + 6 + 1
    assert summary["boxes_after"]["pole"] == 8 and summary["boxes_before"]["pole"] == 2
    assert summary["boxes_after"]["person"] == 5 + 4  # 5 person-only + bus0 repeated 4x


def test_balanced_train_is_deterministic() -> None:
    a, _ = dr.balanced_train(synthetic_scan(), focus=["pole"], person_cap=3, seed=1)
    b, _ = dr.balanced_train(synthetic_scan(), focus=["pole"], person_cap=3, seed=1)
    assert a == b


def test_balanced_files_apply_and_load_in_ultralytics(dataset: Path, drive_zip: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    scan = scan_of(dataset, drive_zip)
    lines, summary = dr.balanced_train(scan, focus=["traffic_cone"], target_images=10)
    done = json.loads(drive_zip.with_suffix(".done").read_text())
    txt, yml = dr.write_balanced(drive_zip.parent, lines, CLASSES, done, summary)
    assert txt.parent == drive_zip.parent and yml.name == "data_balanced.yaml"

    data_yaml = apply_balanced(drive_zip, dataset)
    data = check_det_dataset(str(data_yaml))
    assert data["train"] == str(dataset / "train_balanced.txt")
    ds = YOLODataset(img_path=data["train"], data=data, task="detect", imgsz=64, augment=False,
                     hyp=get_cfg(), batch_size=2)
    assert len(ds) == len(lines) > len(scan.images["train"])  # repeats survive the label cache


def test_apply_balanced_refuses_stale_or_missing(dataset: Path, drive_zip: Path) -> None:
    with pytest.raises(RuntimeError, match="run notebook 03"):
        apply_balanced(drive_zip, dataset)
    scan = scan_of(dataset, drive_zip)
    lines, summary = dr.balanced_train(scan, focus=["traffic_cone"])
    dr.write_balanced(drive_zip.parent, lines, CLASSES, {"zip_bytes": 1}, summary)
    with pytest.raises(RuntimeError, match="older merged.zip"):
        apply_balanced(drive_zip, dataset)
    done = json.loads(drive_zip.with_suffix(".done").read_text())
    dr.write_balanced(drive_zip.parent, lines + ["./images/train/gone.jpg"], CLASSES, done, summary)
    with pytest.raises(RuntimeError, match="gone.jpg"):
        apply_balanced(drive_zip, dataset)
    assert yaml.safe_load((drive_zip.parent / "data_balanced.yaml").read_text())["train"] == "train_balanced.txt"


def test_example_grid(dataset: Path, drive_zip: Path) -> None:
    scan = scan_of(dataset, drive_zip)
    grid = dr.example_grid(dataset, scan, "traffic_cone", seed=0)
    assert grid is not None and grid.shape == (300, 1600, 3)  # 4 cone images -> one row
    empty = dr.Scan(classes=CLASSES, images={s: [] for s in dr.SPLITS}, image_classes={})
    assert dr.example_grid(dataset, empty, "car") is None
