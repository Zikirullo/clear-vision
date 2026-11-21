"""scripts/sync_mobile_assets.py, and the Flutter app's copy of nav_config.json."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sync_mobile_assets as sma  # noqa: E402


def fake_root(tmp_path: Path, run: str, suffixes: tuple[str, ...]) -> Path:
    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True)
    (root / "configs").mkdir()
    shutil.copyfile(ROOT / "models" / "nav_config.json", root / "models" / "nav_config.json")
    shutil.copyfile(ROOT / "configs" / "classes.yaml", root / "configs" / "classes.yaml")
    mobile = root / "exports" / run / "mobile"
    mobile.mkdir(parents=True)
    for suffix in suffixes:
        (mobile / f"clear_vision_{run}{suffix}").write_bytes(b"model")
    return root


def test_sync_copies_models_and_replaces_older_runs(tmp_path: Path) -> None:
    root = fake_root(tmp_path, "v2", (".mlpackage.zip", ".tflite"))
    assets = tmp_path / "assets"
    (assets / "models").mkdir(parents=True)
    (assets / "models" / "clear_vision_budget_v1.mlpackage.zip").write_bytes(b"old")
    (assets / "models" / "README.md").write_text("keep me")
    report = sma.sync("v2", assets=assets, root=root)
    names = sorted(p.name for p in (assets / "models").iterdir())
    assert names == ["README.md", "clear_vision_v2.mlpackage.zip", "clear_vision_v2.tflite", "labels.txt"]
    assert report["removed"] == ["clear_vision_budget_v1.mlpackage.zip"] and report["missing"] == []
    labels = (assets / "models" / "labels.txt").read_text().splitlines()
    assert len(labels) == 20 and labels[0] == "person" and labels[-1] == "fire_hydrant"


def test_missing_models_keep_the_old_one_and_are_reported(tmp_path: Path) -> None:
    root = fake_root(tmp_path, "v3", ())
    assets = tmp_path / "assets"
    (assets / "models").mkdir(parents=True)
    (assets / "models" / "clear_vision_budget_v1.tflite").write_bytes(b"old")
    report = sma.sync("v3", assets=assets, root=root)
    assert (assets / "models" / "clear_vision_budget_v1.tflite").exists()
    assert len(report["missing"]) == 2


def test_flutter_copy_of_nav_config_is_up_to_date() -> None:
    app_copy = ROOT / "mobile_flutter" / "assets" / "nav_config.json"
    assert json.loads(app_copy.read_text()) == json.loads((ROOT / "models" / "nav_config.json").read_text()), (
        "mobile_flutter/assets/nav_config.json is stale: run python scripts/sync_mobile_assets.py --config-only")


def test_flutter_labels_match_classes_yaml() -> None:
    import yaml
    labels = (ROOT / "mobile_flutter" / "assets" / "models" / "labels.txt").read_text().splitlines()
    assert labels == yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]
