"""Tests for scripts/pseudo_label.py with a fake COCO model.

The merged.zip is built by the real merge from tests/assets/merge (classes person, car,
traffic_cone): src_a labels people and cars, src_b only cones.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.merge_datasets import MergeError, merge
from scripts.pseudo_label import (
    BACKUP_FILE,
    CSV_FILE,
    boxes_to_add,
    changes_from_csv,
    class_map,
    format_added_table,
    make_grid,
    pseudo_label,
    source_of,
)
from src.colab_data import prepare_dataset

ASSETS = Path(__file__).parent / "assets" / "merge"
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 9: "traffic light", 10: "fire hydrant"}
PERSON, CAR = 0, 2  # COCO ids
P_OURS, C_OURS = 0, 1  # our ids (person, car, traffic_cone)

A1, A2 = "labels/train/src_a_a1.txt", "labels/train/src_a_a2.txt"
IMG1, IMG2 = "labels/val/src_b_img1_jpg.rf.aaa.txt", "labels/train/src_b_img2_jpg.rf.ccc.txt"
IMG3, IMG4 = "labels/train/src_b_img3_jpg.rf.ddd.txt", "labels/test/src_b_img4_jpg.rf.eee.txt"

# image stem -> (COCO class, confidence, x, y, w, h)
PREDICTIONS = {
    "src_a_a1": [
        (PERSON, 0.90, 0.5, 0.5, 0.2, 0.4),  # same as the labeled person: skipped (overlap)
        (PERSON, 0.90, 0.8, 0.3, 0.1, 0.2),  # a new person: added
        (CAR, 0.80, 0.5, 0.5, 0.2, 0.4),     # overlaps the person, but another class: added
    ],
    "src_b_img2_jpg.rf.ccc": [
        (PERSON, 0.49, 0.7, 0.6, 0.1, 0.3),  # below 0.5: not added
        (CAR, 0.50, 0.3, 0.6, 0.2, 0.1),     # exactly 0.5: added
    ],
    "src_b_img3_jpg.rf.ddd": [(PERSON, 0.95, 0.5, 0.5, 0.3, 0.6)],  # image without labels
    "src_b_img4_jpg.rf.eee": [(PERSON, 0.70, 0.2, 0.5, 0.1, 0.5)],  # test split
    "src_b_img1_jpg.rf.aaa": [(9, 0.99, 0.5, 0.1, 0.05, 0.1)],      # traffic light: not asked for
}


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=float)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self, rows: Sequence[tuple[int, float, float, float, float, float]]) -> None:
        self.cls = FakeTensor([r[0] for r in rows])
        self.conf = FakeTensor([r[1] for r in rows])
        self.xywhn = FakeTensor([r[2:] for r in rows] or np.zeros((0, 4)))

    def __len__(self) -> int:
        return len(self.cls.values)


class FakeResult:
    def __init__(self, rows: Sequence[tuple[int, float, float, float, float, float]]) -> None:
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """Returns PREDICTIONS-style rows by image stem, ignoring predict()'s own `conf`."""

    names = COCO_NAMES
    ckpt_path = "yolo11x.pt"

    def __init__(self, predictions: dict[str, list] | None = None) -> None:
        self.predictions = PREDICTIONS if predictions is None else predictions
        self.seen: list[str] = []

    def predict(self, sources: list[str], **kwargs: Any) -> list[FakeResult]:
        stems = [Path(s).stem for s in sources]
        self.seen += stems
        return [FakeResult(self.predictions.get(stem, [])) for stem in stems]


def no_sync() -> None:
    pass


@pytest.fixture(autouse=True)
def no_font_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)


@pytest.fixture
def drive_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "drive" / "datasets" / "merged.zip"
    merge(raw=ASSETS / "raw", zip_path=zip_path, work=tmp_path / "work",
          classes_path=ASSETS / "classes.yaml", mappings_dir=ASSETS / "mappings", sync=no_sync)
    return zip_path


def run(zip_path: Path, model: FakeModel | None = None, **kwargs: Any) -> dict | None:
    options: dict[str, Any] = dict(
        dataset=zip_path.parent.parent.parent / "content" / "merged", model=model or FakeModel(),
        pseudo_classes=("person", "car"), coco_sources=(), sync=no_sync, check_dataset=False,
    )
    options.update(kwargs)
    return pseudo_label(zip_path, **options)


def zip_text(zip_path: Path, name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name).decode()


def zip_labels(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {n: zf.read(n).decode() for n in zf.namelist() if n.startswith("labels/")}


def lines(text: str) -> list[tuple[int, float, float, float, float]]:
    return [(int(p[0]), *(round(float(v), 3) for v in p[1:])) for p in (ln.split() for ln in text.splitlines())]


# ---------------------------------------------------------------------- pure helpers

def test_boxes_to_add_threshold_and_overlap() -> None:
    existing = [(P_OURS, 0.5, 0.5, 0.2, 0.4)]
    preds = [
        (P_OURS, 0.9, 0.5, 0.5, 0.2, 0.4),    # IoU 1 with the person: skipped
        (P_OURS, 0.9, 0.52, 0.5, 0.2, 0.4),   # IoU ~0.82: skipped
        (P_OURS, 0.9, 0.62, 0.5, 0.2, 0.4),   # IoU ~0.25: added
        (C_OURS, 0.9, 0.5, 0.5, 0.2, 0.4),    # other class: added
        (P_OURS, 0.49, 0.1, 0.1, 0.1, 0.1),   # under the threshold: dropped, not an overlap
    ]
    added, overlaps = boxes_to_add(preds, existing, conf=0.5, iou=0.5)
    assert [(p[0], p[2]) for p in added] == [(P_OURS, 0.62), (C_OURS, 0.5)]
    assert overlaps == 2


def test_source_of_prefers_longest_name() -> None:
    sources = ["rf_scooter", "rf_scooter_bollard", "coco_subset"]
    assert source_of("rf_scooter_bollard_img1", sources) == "rf_scooter_bollard"
    assert source_of("rf_scooter_x", sources) == "rf_scooter"
    assert source_of("unknown_x", sources) is None


def test_class_map_uses_our_ids_and_spelling() -> None:
    classes = ["person", "fire_hydrant", "car"]
    assert class_map(COCO_NAMES, classes, ["person", "fire hydrant", "car"]) == {0: 0, 10: 1, 2: 2}
    with pytest.raises(MergeError, match="dog"):
        class_map(COCO_NAMES, classes, ["dog"])


def test_added_table_has_totals() -> None:
    table = format_added_table({"rf_cone": {"person": 3}, "rf_stairs": {"person": 1, "car": 2}}, ["person", "car"])
    assert table.splitlines()[-1].split() == ["TOTAL", "4", "2", "6"]


# ---------------------------------------------------------------------- the whole step

def test_adds_boxes_and_keeps_originals(drive_zip: Path) -> None:
    before = zip_labels(drive_zip)
    entry = run(drive_zip)
    after = zip_labels(drive_zip)
    assert entry is not None

    # Original lines stay byte for byte at the top of every label file.
    for name, text in before.items():
        assert after[name].startswith(text), name
    # Overlap skip, other class kept, threshold, background image, test split.
    assert lines(after[A1][len(before[A1]):]) == [(P_OURS, 0.8, 0.3, 0.1, 0.2), (C_OURS, 0.5, 0.5, 0.2, 0.4)]
    assert lines(after[IMG2][len(before[IMG2]):]) == [(C_OURS, 0.3, 0.6, 0.2, 0.1)]
    assert lines(after[IMG3]) == [(P_OURS, 0.5, 0.5, 0.3, 0.6)]
    assert lines(after[IMG4][len(before[IMG4]):]) == [(P_OURS, 0.2, 0.5, 0.1, 0.5)]
    # Unchanged files (and the traffic light nobody asked for).
    assert after[A2] == before[A2] and after[IMG1] == before[IMG1]

    assert entry["boxes_added"] == 5 and entry["skipped_overlap"] == 1 and entry["label_files_changed"] == 4
    assert entry["added"] == {"src_a": {"person": 1, "car": 1}, "src_b": {"car": 1, "person": 2}}
    assert entry["added_by_split"]["test"] == {"person": 1}


def test_backup_holds_only_changed_files_with_original_content(drive_zip: Path) -> None:
    before = zip_labels(drive_zip)
    run(drive_zip)
    with zipfile.ZipFile(drive_zip.with_name(BACKUP_FILE)) as zf:
        backup = {n: zf.read(n).decode() for n in zf.namelist()}
    assert set(backup) == {A1, IMG2, IMG3, IMG4}
    assert all(backup[n] == before[n] for n in backup)


def test_csv_and_report_written_next_to_zip(drive_zip: Path) -> None:
    run(drive_zip)
    rows = list(csv.DictReader(drive_zip.with_name(CSV_FILE).open()))
    assert len(rows) == 5
    img3 = next(r for r in rows if "img3" in r["image"])
    assert img3["image"] == "images/train/src_b_img3_jpg.rf.ddd.jpg"
    assert (img3["source"], img3["class"], img3["confidence"]) == ("src_b", "person", "0.9500")

    report = json.loads(drive_zip.with_name("merge_report.json").read_text())
    assert report["pseudo_labels"]["model"] == "yolo11x.pt"
    assert report["sources"]["src_b"]["pseudo_labels_added"] == {"car": 1, "person": 2}
    assert "pseudo_labels_added" in json.loads(zip_text(drive_zip, "merge_report.json"))["sources"]["src_b"]
    assert report["images"] == json.loads(zip_text(drive_zip, "merge_report.json"))["images"]


def test_upload_is_verified_and_done_written_last(drive_zip: Path) -> None:
    done = drive_zip.with_suffix(".done")
    done_at_sync: list[bool] = []
    run(drive_zip, sync=lambda: done_at_sync.append(done.exists()))
    # 1st flush: backup + CSV only (merged.zip untouched, still done). 2nd: new zip, done deleted first.
    assert done_at_sync == [True, False]
    info = json.loads(done.read_text())
    assert info["zip_bytes"] == drive_zip.stat().st_size
    with zipfile.ZipFile(drive_zip) as zf:
        assert zf.testzip() is None
        assert not [n for n in zf.namelist() if n.endswith((".prepared", ".cache"))]


def test_dataset_still_passes_prepare_and_check(drive_zip: Path, tmp_path: Path) -> None:
    run(drive_zip)
    data_yaml = prepare_dataset(drive_zip, tmp_path / "fresh", check=True)  # runs check_det_dataset
    assert data_yaml.exists()
    # The unpack used for labeling is marked as matching the new zip, so it is reused.
    reused = prepare_dataset(drive_zip, drive_zip.parent.parent.parent / "content" / "merged", check=True)
    assert reused.exists()


def test_skips_coco_sources(drive_zip: Path) -> None:
    before = zip_labels(drive_zip)
    model = FakeModel()
    entry = run(drive_zip, model=model, coco_sources=("src_a",))
    assert entry is not None
    assert not [s for s in model.seen if s.startswith("src_a_")]
    assert zip_labels(drive_zip)[A1] == before[A1]
    assert set(entry["added"]) == {"src_b"}


def test_rerun_skips_without_force(drive_zip: Path) -> None:
    run(drive_zip)
    zip_bytes = drive_zip.read_bytes()
    model = FakeModel()
    assert run(drive_zip, model=model) is None
    assert model.seen == []
    assert drive_zip.read_bytes() == zip_bytes


def test_force_restores_originals_before_labeling_again(drive_zip: Path) -> None:
    before = zip_labels(drive_zip)
    run(drive_zip)
    # Second run finds only the person in img3 (same box as last time).
    entry = run(drive_zip, model=FakeModel({"src_b_img3_jpg.rf.ddd": PREDICTIONS["src_b_img3_jpg.rf.ddd"]}), force=True)
    after = zip_labels(drive_zip)

    assert entry is not None and entry["added"] == {"src_b": {"person": 1}}
    assert after[A1] == before[A1] and after[IMG2] == before[IMG2] and after[IMG4] == before[IMG4]
    assert lines(after[IMG3]) == [(P_OURS, 0.5, 0.5, 0.3, 0.6)]  # once, not twice
    report = json.loads(drive_zip.with_name("merge_report.json").read_text())
    assert "pseudo_labels_added" not in report["sources"]["src_a"]
    # The backup still covers every file the first run changed.
    with zipfile.ZipFile(drive_zip.with_name(BACKUP_FILE)) as zf:
        assert set(zf.namelist()) == {A1, IMG2, IMG3, IMG4}
        assert zf.read(A1).decode() == before[A1]


def test_force_without_backup_fails(drive_zip: Path) -> None:
    run(drive_zip)
    drive_zip.with_name(BACKUP_FILE).unlink()
    with pytest.raises(MergeError, match="cannot be restored"):
        run(drive_zip, force=True)


def test_failed_backup_upload_leaves_zip_alone(drive_zip: Path) -> None:
    zip_bytes = drive_zip.read_bytes()
    backup = drive_zip.with_name(BACKUP_FILE)

    def lose_backup() -> None:  # Drive "lost" the end of the backup
        backup.write_bytes(backup.read_bytes()[:-10])

    with pytest.raises(MergeError, match="NOT changed"):
        run(drive_zip, sync=lose_backup)
    assert drive_zip.read_bytes() == zip_bytes
    assert drive_zip.with_suffix(".done").exists()
    report = json.loads(drive_zip.with_name("merge_report.json").read_text())
    assert "pseudo_labels" not in report


def test_crash_mid_run_forces_fresh_unpack(drive_zip: Path) -> None:
    class Boom(FakeModel):
        def predict(self, sources: list[str], **kwargs: Any) -> list[FakeResult]:
            raise RuntimeError("GPU out of memory")

    with pytest.raises(RuntimeError):
        run(drive_zip, model=Boom())
    dataset = drive_zip.parent.parent.parent / "content" / "merged"
    assert not (dataset / ".prepared").exists()


def test_grid_draws_changed_images(drive_zip: Path, tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.jpg"
    run(drive_zip, grid_path=grid_path)
    assert grid_path.exists()
    assert make_grid(tmp_path, [], ["person"]) is None


def test_changes_rebuilt_from_csv_for_a_later_grid(drive_zip: Path) -> None:
    run(drive_zip)
    dataset = drive_zip.parent.parent.parent / "content" / "merged"
    changes = {c.label: c for c in changes_from_csv(dataset, drive_zip.with_name(CSV_FILE), ["person", "car", "traffic_cone"])}
    assert set(changes) == {A1, IMG2, IMG3, IMG4}
    a1 = changes[A1]
    assert [b[0] for b in a1.original_boxes] == [P_OURS, C_OURS]  # the two labeled boxes
    assert [p[0] for p in a1.added] == [P_OURS, C_OURS]  # the two added ones
    assert changes[IMG3].original_boxes == []
    assert make_grid(dataset, list(changes.values()), ["person", "car", "traffic_cone"]).shape == (360, 4 * 480, 3)
