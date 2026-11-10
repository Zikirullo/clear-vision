"""Tests for scripts/merge_datasets.py on the tiny fake datasets in tests/assets/merge/.

src_a: FiftyOne layout (images/<split>), classes Person and Car, one background image.
src_b: Roboflow layout (<split>/images), classes bump (dropped, reviewed) and cone. It has
an augmented pair (img1_jpg.rf.aaa / .bbb), an own test split, a polygon label and an
exact copy of src_a's a1.jpg.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.merge_datasets import (
    REPO,
    MergeError,
    Record,
    assign_splits,
    group_key,
    load_classes,
    load_mappings,
    mask_to_boxes,
    merge,
    parse_label_text,
)

ASSETS = Path(__file__).parent / "assets" / "merge"
CLASSES = ["person", "car", "traffic_cone"]


def run_merge(tmp_path: Path, raw: Path = ASSETS / "raw", mappings: Path = ASSETS / "mappings") -> dict:
    return merge(
        raw=raw,
        out=tmp_path / "out",
        work=tmp_path / "work",
        classes_path=ASSETS / "classes.yaml",
        mappings_dir=mappings,
    )


def labels_of(out: Path) -> dict[str, tuple[str, list[str]]]:
    """Merged label files: {stem: (split, lines)}."""
    return {
        p.stem: (p.parent.name, p.read_text().splitlines())
        for p in (out / "labels").rglob("*.txt")
    }


@pytest.fixture
def merged(tmp_path: Path) -> tuple[Path, dict]:
    report = run_merge(tmp_path)
    return tmp_path / "out", report


# ---------------------------------------------------------------------- end to end


def test_writes_data_yaml_report_and_done_marker(merged: tuple[Path, dict]) -> None:
    out, report = merged
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == CLASSES
    assert data["nc"] == 3
    assert Path(data["path"]) == out.absolute()
    assert (out / ".done").exists()
    assert json.loads((out / "merge_report.json").read_text())["images"] == report["images"]
    assert not (out / "_staging").exists()


def test_filenames_prefixed_and_duplicate_removed(merged: tuple[Path, dict]) -> None:
    out, report = merged
    images = sorted(p.name for p in (out / "images").rglob("*.jpg"))
    assert all(name.startswith(("src_a_", "src_b_")) for name in images)
    assert len(images) == 3 + 6 - 1  # src_b's copy of a1.jpg is gone
    assert "src_b_dup_jpg.rf.fff.jpg" not in images
    assert report["sources"]["src_b"]["duplicates_removed"] == 1
    assert sum(report["images"].values()) == 8


def test_labels_rewritten_with_target_ids(merged: tuple[Path, dict]) -> None:
    labels = labels_of(merged[0])
    a1 = labels["src_a_a1"][1]
    assert [line.split()[0] for line in a1] == ["0", "1"]  # Person -> person, Car -> car
    assert a1[0] == "0 0.500000 0.500000 0.200000 0.400000"
    # cone -> traffic_cone (2); the bump box is dropped
    assert [line.split()[0] for line in labels["src_b_img1_jpg.rf.bbb"][1]] == ["2"]
    assert labels["src_b_img3_jpg.rf.ddd"][1] == []  # only a bump box
    assert labels["src_a_a3"][1] == []  # background image without a label file


def test_polygon_label_becomes_box(merged: tuple[Path, dict]) -> None:
    line = labels_of(merged[0])["src_b_img2_jpg.rf.ccc"][1][0]
    assert line == "2 0.200000 0.250000 0.200000 0.300000"


def test_augmented_copies_share_split_and_own_test_kept(merged: tuple[Path, dict]) -> None:
    labels = labels_of(merged[0])
    assert labels["src_b_img1_jpg.rf.aaa"][0] == labels["src_b_img1_jpg.rf.bbb"][0]
    assert labels["src_b_img4_jpg.rf.eee"][0] == "test"


def test_report_counts_boxes_and_dropped(merged: tuple[Path, dict]) -> None:
    _, report = merged
    total = {c: sum(report["boxes"][s][c] for s in ("train", "val", "test")) for c in CLASSES}
    assert total == {"person": 1, "car": 2, "traffic_cone": 4}
    assert report["sources"]["src_b"]["dropped_boxes"] == {"bump": 2}
    assert report["sources"]["src_a"]["missing_label_files"] == 1
    assert set(report["low_train_classes"]) == set(CLASSES)  # all under 300


def test_review_images_saved(merged: tuple[Path, dict]) -> None:
    out, report = merged
    saved = sorted(p.name for p in (out / "review" / "bump").iterdir())
    assert saved == ["src_b_img1_jpg.rf.bbb.jpg", "src_b_img3_jpg.rf.ddd.jpg"]
    assert report["review_samples"] == {"src_b/bump": 2}


def test_ultralytics_accepts_dataset(merged: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)  # avoid a font download
    data = data_utils.check_det_dataset(str(merged[0] / "data.yaml"))
    assert data["nc"] == 3
    assert list(data["names"].values()) == CLASSES


def test_rerun_skips_when_done(tmp_path: Path) -> None:
    assert run_merge(tmp_path) is not None
    assert run_merge(tmp_path) is None


# ---------------------------------------------------------------------- failures


def test_unmapped_class_fails_before_copying(tmp_path: Path) -> None:
    mappings = tmp_path / "mappings"
    shutil.copytree(ASSETS / "mappings", mappings)
    (mappings / "src_b.yaml").write_text("folder: src_b\nmap:\n  cone: traffic_cone\n")
    with pytest.raises(MergeError, match=r"\['bump'\].*not in configs/mappings/src_b.yaml"):
        run_merge(tmp_path, mappings=mappings)
    assert not (tmp_path / "work").exists()


def test_mapping_typo_fails(tmp_path: Path) -> None:
    mappings = tmp_path / "mappings"
    shutil.copytree(ASSETS / "mappings", mappings)
    (mappings / "src_a.yaml").write_text("folder: src_a\nmap:\n  Person: person\n  Car: car\n  Cat: drop\n")
    with pytest.raises(MergeError, match="'Cat'"):
        run_merge(tmp_path, mappings=mappings)


def test_folder_without_mapping_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(ASSETS / "raw", raw)
    (raw / "extra_dataset").mkdir()
    with pytest.raises(MergeError, match="extra_dataset"):
        run_merge(tmp_path, raw=raw)


def test_labels_in_unknown_layout_fail(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(ASSETS / "raw", raw)
    shutil.move(raw / "src_a" / "labels", raw / "src_a" / "annotations")
    with pytest.raises(MergeError, match="3 of 3 images have no label file"):
        run_merge(tmp_path, raw=raw)


def test_refuses_to_delete_foreign_folder(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "precious.txt").write_text("keep me")
    with pytest.raises(MergeError, match="Refusing to delete"):
        run_merge(tmp_path)
    assert (tmp_path / "work" / "precious.txt").exists()


# ---------------------------------------------------------------------- units


def test_group_key() -> None:
    assert group_key("IMG_0042_jpg.rf.3f9a1c") == "IMG_0042_jpg"
    assert group_key("000000011197") == "000000011197"


def test_parse_label_text_skips_invalid_lines() -> None:
    boxes, invalid = parse_label_text("0 0.5 0.5 0.2 0.2\n0 0.5\n1 0.5 0.5 0 0.3\n\n", 2, Path("x.txt"))
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx((0, 0.5, 0.5, 0.2, 0.2))
    assert invalid == 2


def test_parse_label_text_unknown_class_id_fails() -> None:
    with pytest.raises(MergeError, match="class id 5"):
        parse_label_text("5 0.5 0.5 0.2 0.2\n", 2, Path("x.txt"))


def test_assign_splits_grouped_ratios_and_seed() -> None:
    def make() -> list[Record]:
        records = []
        for i in range(400):
            for copy in ("a", "b"):  # two augmented copies per group
                records.append(Record("src", Path(f"g{i}_{copy}.jpg"), f"src/g{i}", own_test=i < 20, boxes=[]))
        return records

    first, second = make(), make()
    assign_splits(first, (0.75, 0.15, 0.10), seed=42)
    assign_splits(second, (0.75, 0.15, 0.10), seed=42)
    assert [r.split for r in first] == [r.split for r in second]
    by_group = {}
    for r in first:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert all(by_group[f"src/g{i}"] == {"test"} for i in range(20))
    counts = {s: sum(1 for v in by_group.values() if v == {s}) for s in ("train", "val", "test")}
    assert counts == {"train": 300, "val": 60, "test": 40}


def test_mask_to_boxes_converts_wanted_and_skips_small() -> None:
    mask = np.zeros((100, 200), np.uint16)
    mask[10:60, 20:40] = 5 * 256 + 1  # wanted label 5, instance 1: 20 x 50 px
    mask[70:80, 100:105] = 5 * 256 + 2  # wanted but 5 x 10 px: too small
    mask[0:50, 150:200] = 7 * 256 + 1  # unwanted label
    boxes, small = mask_to_boxes(mask, {5: 3}, min_size=12)
    assert boxes == [(3, 30 / 200, 35 / 100, 20 / 200, 50 / 100)]
    assert small == 1


def test_real_mapping_files_are_valid() -> None:
    classes = load_classes(REPO / "configs" / "classes.yaml")
    assert classes[:3] == ["person", "bicycle", "car"] and len(classes) == 20
    mappings = {m.short_name: m for m in load_mappings(REPO / "configs" / "mappings", classes)}
    assert set(mappings) >= {
        "coco_subset", "openimages", "mendeley", "rf_bollard", "rf_cone", "rf_crosswalk",
        "rf_manhole", "rf_pole_curb_stairs", "rf_pothole", "rf_scooter_bollard", "rf_stairs",
    }
    assert mappings["rf_crosswalk"].classes["2"] == "drop"
    assert mappings["rf_crosswalk"].review == ("pedestrian-crossing",)
    assert mappings["mapillary"].format == "mapillary"
