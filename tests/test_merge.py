"""Tests for scripts/merge_datasets.py on the tiny fake datasets in tests/assets/merge/.

src_a: FiftyOne layout (images/<split>), classes Person and Car, one background image.
src_b: Roboflow layout (<split>/images), classes bump (dropped, reviewed) and cone. It has
an augmented pair (img1_jpg.rf.aaa / .bbb), an own test split, a polygon label and an
exact copy of src_a's a1.jpg.
new_raw/src_c (a second raw folder, like /content/raw): a 2000x1000 PNG augmented pair,
a small JPEG, a copy of src_a's a2.jpg and an unlisted class "junk".

The merge writes <tmp>/drive/datasets/merged.zip; `sync` (the Drive flush) is a no-op
or a function that damages the upload.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
import yaml

from scripts.merge_datasets import (
    LOCAL_DATASET,
    REPO,
    Mapping,
    MergeError,
    Record,
    append,
    assign_splits,
    changed_classes,
    group_key,
    load_classes,
    load_mappings,
    mask_to_boxes,
    merge,
    parse_label_text,
)

ASSETS = Path(__file__).parent / "assets" / "merge"
CLASSES = ["person", "car", "traffic_cone"]
SPLITS = ("train", "val", "test")


def no_sync() -> None:
    pass


def zip_of(tmp_path: Path) -> Path:
    return tmp_path / "drive" / "datasets" / "merged.zip"


def run_merge(
    tmp_path: Path, raw: Path | list[Path] = ASSETS / "raw", mappings: Path = ASSETS / "mappings",
    sync: Callable[[], None] = no_sync, **kwargs,
) -> dict | None:
    return merge(
        raw=raw,
        zip_path=zip_of(tmp_path),
        work=tmp_path / "work",
        classes_path=ASSETS / "classes.yaml",
        mappings_dir=mappings,
        sync=sync,
        **kwargs,
    )


def unpack(tmp_path: Path) -> Path:
    """Extract merged.zip, as training would see it."""
    dest = tmp_path / "unpacked"
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(zip_of(tmp_path)) as zf:
        zf.extractall(dest)
    return dest


def labels_of(dataset: Path) -> dict[str, tuple[str, list[str]]]:
    """Merged label files: {stem: (split, lines)}."""
    return {p.stem: (p.parent.name, p.read_text().splitlines()) for p in (dataset / "labels").rglob("*.txt")}


def boxes_on_disk(dataset: Path) -> dict[str, dict[str, int]]:
    """Count boxes per split and class straight from the merged label files."""
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for path in (dataset / "labels").rglob("*.txt"):
        for line in path.read_text().splitlines():
            counts[path.parent.name][CLASSES[int(line.split()[0])]] += 1
    return counts


def with_src_c(tmp_path: Path, src_b: str | None = None) -> Path:
    """Base mappings + src_c, optionally with a changed src_b mapping file."""
    mappings = tmp_path / "mappings_c"
    shutil.copytree(ASSETS / "mappings", mappings)
    shutil.copy(ASSETS / "append_mappings" / "src_c.yaml", mappings)
    if src_b is not None:
        (mappings / "src_b.yaml").write_text(src_b)
    return mappings


@pytest.fixture
def merged(tmp_path: Path) -> tuple[Path, dict]:
    report = run_merge(tmp_path)
    return unpack(tmp_path), report


# ---------------------------------------------------------------------- full merge


def test_writes_verified_zip_report_and_done(merged: tuple[Path, dict], tmp_path: Path) -> None:
    dataset, report = merged
    drive = zip_of(tmp_path).parent
    done = json.loads((drive / "merged.done").read_text())
    assert done["zip_bytes"] == zip_of(tmp_path).stat().st_size
    assert done["images"] == report["images"]
    assert json.loads((drive / "merge_report.json").read_text()) == report
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    assert data["names"] == CLASSES and data["nc"] == 3
    assert data["path"] == LOCAL_DATASET
    assert (dataset / "merge_report.json").exists() and (dataset / "md5_index.json").exists()
    assert not (dataset / "review").exists()  # review/ is next to the zip, not in it
    assert not (tmp_path / "work").exists()
    with zipfile.ZipFile(zip_of(tmp_path)) as zf:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())


def test_filenames_prefixed_and_duplicate_removed(merged: tuple[Path, dict]) -> None:
    dataset, report = merged
    images = sorted(p.name for p in (dataset / "images").rglob("*.jpg"))
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
    dataset, report = merged
    total = {c: sum(report["boxes"][s][c] for s in SPLITS) for c in CLASSES}
    assert total == {"person": 1, "car": 2, "traffic_cone": 4}
    assert report["boxes"] == boxes_on_disk(dataset)
    assert report["sources"]["src_b"]["dropped_boxes"] == {"bump": 2}
    assert report["sources"]["src_a"]["missing_label_files"] == 1
    assert set(report["low_train_classes"]) == set(CLASSES)  # all under 300


def test_review_images_saved_next_to_zip(merged: tuple[Path, dict], tmp_path: Path) -> None:
    _, report = merged
    saved = sorted(p.name for p in (zip_of(tmp_path).parent / "review" / "bump").iterdir())
    assert saved == ["src_b_img1_jpg.rf.bbb.jpg", "src_b_img3_jpg.rf.ddd.jpg"]
    assert report["review_samples"] == {"src_b/bump": 2}


def test_full_merge_resizes_and_includes_second_raw_folder(tmp_path: Path) -> None:
    report = run_merge(tmp_path, raw=[ASSETS / "new_raw", ASSETS / "raw"], mappings=with_src_c(tmp_path))
    dataset = unpack(tmp_path)
    big = next((dataset / "images").rglob("src_c_big_png.rf.aaa.jpg"))
    assert cv2.imread(str(big)).shape[:2] == (640, 1280)  # 2000x1000 PNG -> 1280 px JPEG
    small = next((dataset / "images").rglob("src_a_a1.jpg"))
    assert small.read_bytes() == (ASSETS / "raw/src_a/images/train/a1.jpg").read_bytes()
    assert "src_c" in report["sources"]
    assert report["sources"]["src_c"]["duplicates_removed"] == 1  # copy of a2.jpg
    # Recorded (solid-color test PNGs are tiny, so "smaller after resize" isn't checked here).
    assert report["sizes"]["images_merged_bytes"] > 0 and report["sizes"]["images_source_bytes"] > 0
    assert report["sizes"]["max_side"] == 1280
    assert all(p.suffix == ".jpg" for p in (dataset / "images").rglob("*.*"))


def test_ultralytics_accepts_dataset(merged: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    import ultralytics.data.utils as data_utils

    dataset, _ = merged
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    data["path"] = str(dataset)
    (dataset / "data.yaml").write_text(yaml.safe_dump(data))
    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)  # avoid a font download
    result = data_utils.check_det_dataset(str(dataset / "data.yaml"))
    assert list(result["names"].values()) == CLASSES


def test_rerun_skips_when_done(tmp_path: Path) -> None:
    assert run_merge(tmp_path) is not None
    assert run_merge(tmp_path) is None
    assert run_merge(tmp_path, force=True) is not None


# ---------------------------------------------------------------------- Drive safety


def test_stops_when_drive_is_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10**9, 10**9 - 1000, 1000))
    with pytest.raises(MergeError, match="Not enough space.*Nothing was written"):
        run_merge(tmp_path)
    assert not zip_of(tmp_path).parent.exists()


def truncate_upload(tmp_path: Path) -> Callable[[], None]:
    def sync() -> None:  # Drive "lost" the end of the file
        path = zip_of(tmp_path)
        path.write_bytes(path.read_bytes()[:-100])
    return sync


def corrupt_upload(tmp_path: Path) -> Callable[[], None]:
    def sync() -> None:  # same size, one damaged byte inside an image
        path = zip_of(tmp_path)
        with zipfile.ZipFile(path) as zf:
            info = next(i for i in zf.infolist() if i.filename.endswith(".jpg"))
        data = bytearray(path.read_bytes())
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + 10
        data[offset] ^= 0xFF
        path.write_bytes(bytes(data))
    return sync


def test_incomplete_upload_is_not_marked_done(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="Upload incomplete"):
        run_merge(tmp_path, sync=truncate_upload(tmp_path))
    assert not (zip_of(tmp_path).parent / "merged.done").exists()


def test_corrupt_upload_is_not_marked_done(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="corrupt"):
        run_merge(tmp_path, sync=corrupt_upload(tmp_path))
    assert not (zip_of(tmp_path).parent / "merged.done").exists()


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


def test_missing_source_folder_fails(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="src_c: folder src_c not found"):
        run_merge(tmp_path, mappings=with_src_c(tmp_path))  # src_c only exists in new_raw


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
    by_group: dict[str, set[str]] = {}
    for r in first:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert all(by_group[f"src/g{i}"] == {"test"} for i in range(20))
    counts = {s: sum(1 for v in by_group.values() if v == {s}) for s in SPLITS}
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
        "rf_sidewalk", "rf_barricade",
    }
    assert mappings["rf_crosswalk"].classes["2"] == "drop"
    assert mappings["rf_crosswalk"].review == ("pedestrian-crossing",)
    assert mappings["mapillary"].format == "mapillary"


# ---------------------------------------------------------------------- append
#
# Base: the full merge of src_a + src_b (fixture `merged`). Append: src_c from new_raw/.


def run_append(tmp_path: Path, sources: list[str], mappings: Path, **kwargs) -> dict | None:
    kwargs.setdefault("sync", no_sync)
    return append(
        raw=[ASSETS / "new_raw", ASSETS / "raw"],
        zip_path=zip_of(tmp_path),
        new_sources=sources,
        work=tmp_path / "append_work",
        classes_path=ASSETS / "classes.yaml",
        mappings_dir=mappings,
        **kwargs,
    )


def test_append_adds_new_source(merged: tuple[Path, dict], tmp_path: Path) -> None:
    _, before = merged
    report = run_append(tmp_path, ["src_c"], with_src_c(tmp_path))
    dataset = unpack(tmp_path)
    names = {p.name for p in (dataset / "images").rglob("*.*")}
    assert {"src_c_big_png.rf.aaa.jpg", "src_c_big_png.rf.bbb.jpg", "src_c_small_jpg.rf.ddd.jpg"} <= names
    assert "src_c_copy_jpg.rf.ccc.jpg" not in names  # same bytes as merged src_a_a2.jpg
    src_c = report["sources"]["src_c"]
    assert src_c["duplicates_removed"] == 1
    assert src_c["dropped_boxes"] == {"junk": 1}
    assert sum(src_c["images"].values()) == 3
    assert report["sources"]["src_a"] == before["sources"]["src_a"]  # untouched
    assert sum(report["images"].values()) == sum(before["images"].values()) + 3
    assert report["boxes"] == boxes_on_disk(dataset)  # totals recomputed
    assert report["appends"][-1]["sources"] == ["src_c"]
    drive = zip_of(tmp_path).parent
    assert json.loads((drive / "merge_report.json").read_text()) == report
    assert json.loads((dataset / "merge_report.json").read_text()) == report
    assert json.loads((drive / "merged.done").read_text())["zip_bytes"] == zip_of(tmp_path).stat().st_size
    assert (drive / "review" / "bump").exists()  # the earlier review images are kept
    assert not (tmp_path / "append_work").exists()


def test_append_resizes_and_keeps_pair_together(merged: tuple[Path, dict], tmp_path: Path) -> None:
    run_append(tmp_path, ["src_c"], with_src_c(tmp_path))
    dataset = unpack(tmp_path)
    big = next((dataset / "images").rglob("src_c_big_png.rf.aaa.jpg"))
    assert cv2.imread(str(big)).shape[:2] == (640, 1280)
    small = next((dataset / "images").rglob("src_c_small_jpg.rf.ddd.jpg"))
    assert small.read_bytes() == (ASSETS / "new_raw/src_c/valid/images/small_jpg.rf.ddd.jpg").read_bytes()
    assert len({p.parent.name for p in (dataset / "labels").rglob("src_c_big_png.rf.*.txt")}) == 1


def test_append_updates_md5_index(merged: tuple[Path, dict], tmp_path: Path) -> None:
    run_append(tmp_path, ["src_c"], with_src_c(tmp_path))
    assert len(json.loads((unpack(tmp_path) / "md5_index.json").read_text())) == 8 + 3


def test_append_rerun_skips_already_merged_source(merged: tuple[Path, dict], tmp_path: Path) -> None:
    mappings = with_src_c(tmp_path)
    assert run_append(tmp_path, ["src_c"], mappings) is not None
    assert run_append(tmp_path, ["src_c"], mappings) is None


def test_append_stops_when_drive_is_full(
    merged: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_bytes = zip_of(tmp_path).read_bytes()
    done = (zip_of(tmp_path).parent / "merged.done").read_text()
    # The old zip's space counts as free (it is replaced), so make even that too little.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10**9, 10**9 - 1000, 1000))
    with pytest.raises(MergeError, match="Not enough space"):
        run_append(tmp_path, ["src_c"], with_src_c(tmp_path))
    assert zip_of(tmp_path).read_bytes() == zip_bytes
    assert (zip_of(tmp_path).parent / "merged.done").read_text() == done


def test_append_incomplete_upload_is_not_marked_done(merged: tuple[Path, dict], tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="Upload incomplete"):
        run_append(tmp_path, ["src_c"], with_src_c(tmp_path), sync=truncate_upload(tmp_path))
    assert not (zip_of(tmp_path).parent / "merged.done").exists()


def test_append_unmapped_class_fails(merged: tuple[Path, dict], tmp_path: Path) -> None:
    mappings = with_src_c(tmp_path)
    (mappings / "src_c.yaml").write_text("folder: src_c\nmap:\n  Cone: traffic_cone\n  walker: person\n")
    with pytest.raises(MergeError, match=r"\['junk'\]"):
        run_append(tmp_path, ["src_c"], mappings)


def test_append_needs_verified_merge(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="Run a full merge first"):
        run_append(tmp_path, ["src_c"], with_src_c(tmp_path))


def test_append_remaps_changed_mapping_of_merged_source(merged: tuple[Path, dict], tmp_path: Path) -> None:
    before_dataset, before = merged
    before_labels = labels_of(before_dataset)
    # bump was dropped in the merge; now it maps to car (like pedestrian-crossing -> crosswalk).
    mappings = with_src_c(tmp_path, "folder: src_b\nmap:\n  cone: traffic_cone\n  bump: car\nreview:\n  - bump\n")
    report = run_append(tmp_path, [], mappings)
    dataset = unpack(tmp_path)
    labels = labels_of(dataset)
    assert labels["src_b_img3_jpg.rf.ddd"][1] == ["1 0.500000 0.500000 0.900000 0.300000"]
    assert sorted(line.split()[0] for line in labels["src_b_img1_jpg.rf.bbb"][1]) == ["1", "2"]
    assert report["sources"]["src_b"]["dropped_boxes"] == {}
    assert report["sources"]["src_b"]["mapping"]["bump"] == "car"
    assert report["sources"]["src_b"]["duplicates_removed"] == 1
    assert report["appends"][-1]["remapped"] == {"src_b": ["bump"]}
    car = lambda r: sum(r["boxes"][s]["car"] for s in SPLITS)  # noqa: E731
    assert car(report) == car(before) + 2
    assert report["boxes"] == boxes_on_disk(dataset)
    assert labels["src_a_a1"] == before_labels["src_a_a1"]  # other sources untouched
    assert run_append(tmp_path, [], mappings) is None  # mapping recorded: nothing left to do


def test_changed_classes_for_reports_without_recorded_mapping() -> None:
    m = Mapping("rf_crosswalk", "rf_crosswalk", {"crosswalk": "crosswalk", "pedestrian-crossing": "crosswalk", "2": "drop"})
    old_entry = {"dropped_boxes": {"pedestrian-crossing": 40, "2": 3}}
    assert changed_classes(old_entry, m) == ["pedestrian-crossing"]
    recorded = {"mapping": {"crosswalk": "crosswalk", "pedestrian-crossing": "drop", "2": "drop"}}
    assert changed_classes(recorded, m) == ["pedestrian-crossing"]
    assert changed_classes({"mapping": dict(m.classes)}, m) == []
