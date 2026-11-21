"""Copy the navigation config and a trained run's phone models into the Flutter app.

    python scripts/sync_mobile_assets.py                 # run budget_v1 (default)
    python scripts/sync_mobile_assets.py --run v2        # swap in another model
    python scripts/sync_mobile_assets.py --config-only   # only models/nav_config.json

Copies into mobile_flutter/assets/:
- nav_config.json        <- models/nav_config.json (numbers and phrases; see docs/navigation_spec.md)
- models/labels.txt      <- configs/classes.yaml names (the app checks the model's names against it)
- models/clear_vision_<run>.mlpackage.zip  (iOS)   <- exports/<run>/mobile/
- models/clear_vision_<run>.tflite         (Android) <- exports/<run>/mobile/

Older clear_vision_*.mlpackage.zip / *.tflite in assets/models/ are removed, so the app
(which loads whichever one is there) always uses the run you synced. The model files are
gitignored (download them from MyDrive/clear_vision/exports/<run>/mobile/ into
exports/<run>/mobile/ first); nav_config.json and labels.txt are committed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP_ASSETS = ROOT / "mobile_flutter" / "assets"
MODEL_SUFFIXES = (".mlpackage.zip", ".tflite")

log = logging.getLogger("sync_mobile_assets")


def model_files(run: str, exports: Path = ROOT / "exports") -> dict[str, Path]:
    """The two plugin models of a run: {suffix: path} (missing files left out)."""
    mobile = exports / run / "mobile"
    found = {}
    for suffix in MODEL_SUFFIXES:
        path = mobile / f"clear_vision_{run}{suffix}"
        if path.exists():
            found[suffix] = path
    return found


def remove_old_models(models_dir: Path, keep: Sequence[str]) -> list[str]:
    removed = []
    for path in sorted(models_dir.glob("clear_vision_*")):
        if path.name.endswith(MODEL_SUFFIXES) and path.name not in keep:
            path.unlink()
            removed.append(path.name)
    return removed


def sync(run: str, assets: Path = APP_ASSETS, root: Path = ROOT, config_only: bool = False) -> dict[str, list[str]]:
    """Copy config, labels and models; returns what was copied, removed and missing."""
    models_dir = assets / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "models" / "nav_config.json", assets / "nav_config.json")
    classes = yaml.safe_load((root / "configs" / "classes.yaml").read_text())["names"]
    (models_dir / "labels.txt").write_text("".join(f"{c}\n" for c in classes))
    report: dict[str, list[str]] = {"copied": ["nav_config.json", "models/labels.txt"], "removed": [], "missing": []}
    if config_only:
        return report
    found = model_files(run, root / "exports")
    for suffix in MODEL_SUFFIXES:
        if suffix not in found:
            report["missing"].append(f"exports/{run}/mobile/clear_vision_{run}{suffix}")
    if not found:  # keep whatever model is there rather than leaving the app with none
        return report
    report["removed"] = remove_old_models(models_dir, keep=[p.name for p in found.values()])
    for path in found.values():
        shutil.copyfile(path, models_dir / path.name)
        report["copied"].append(f"models/{path.name}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="budget_v1", help="training run name (exports/<run>/mobile/)")
    ap.add_argument("--config-only", action="store_true", help="copy nav_config.json and labels.txt only")
    args = ap.parse_args(argv)
    report = sync(args.run, config_only=args.config_only)
    for name in report["copied"]:
        log.info("copied   mobile_flutter/assets/%s", name)
    for name in report["removed"]:
        log.info("removed  mobile_flutter/assets/models/%s (older model)", name)
    for name in report["missing"]:
        log.warning("MISSING  %s: download it from MyDrive/clear_vision/%s", name, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
