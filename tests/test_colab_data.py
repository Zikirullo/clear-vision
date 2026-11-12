"""Tests for src/colab_data.py prepare_dataset() on a merged.zip built from the fake datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.merge_datasets import merge
from src.colab_data import prepare_dataset

ASSETS = Path(__file__).parent / "assets" / "merge"


@pytest.fixture
def drive_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "drive" / "datasets" / "merged.zip"
    merge(
        raw=ASSETS / "raw", zip_path=zip_path, work=tmp_path / "work",
        classes_path=ASSETS / "classes.yaml", mappings_dir=ASSETS / "mappings", sync=lambda: None,
    )
    return zip_path


@pytest.fixture(autouse=True)
def no_font_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)


def test_prepare_unpacks_and_checks(drive_zip: Path, tmp_path: Path) -> None:
    dest = tmp_path / "content" / "datasets" / "merged"
    data_yaml = prepare_dataset(drive_zip, dest)
    assert data_yaml == dest / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    assert data["path"] == str(dest.absolute())
    assert data["names"] == ["person", "car", "traffic_cone"]
    assert sum(1 for _ in (dest / "images").rglob("*.jpg")) == 8
    assert not (dest.parent / "merged.zip").exists()  # the local copy is removed


def test_prepare_reuses_earlier_unpack(drive_zip: Path, tmp_path: Path) -> None:
    dest = tmp_path / "local"
    prepare_dataset(drive_zip, dest)
    (dest / "sentinel").write_text("still here")
    prepare_dataset(drive_zip, dest)
    assert (dest / "sentinel").exists()


def test_prepare_requires_verified_zip(drive_zip: Path, tmp_path: Path) -> None:
    drive_zip.with_suffix(".done").unlink()
    with pytest.raises(RuntimeError, match="never verified"):
        prepare_dataset(drive_zip, tmp_path / "local")


def test_prepare_detects_image_count_mismatch(drive_zip: Path, tmp_path: Path) -> None:
    report_path = drive_zip.with_name("merge_report.json")
    report = json.loads(report_path.read_text())
    report["images"]["val"] += 5  # e.g. files lost on the way
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="Image count mismatch"):
        prepare_dataset(drive_zip, tmp_path / "local", check=False)


def test_prepare_detects_size_mismatch(drive_zip: Path, tmp_path: Path) -> None:
    done = drive_zip.with_suffix(".done")
    info = json.loads(done.read_text())
    info["zip_bytes"] += 1
    done.write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match="merged.done says"):
        prepare_dataset(drive_zip, tmp_path / "local", check=False)
