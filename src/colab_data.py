"""Load the merged dataset in Colab from the verified merged.zip on Drive.

Training never reads the dataset folder on Drive: Drive is slow with many small files,
and a folder that looked complete in Colab's cache was missing most of its files.
prepare_dataset() copies the single zip to the Colab disk, unpacks it and checks it.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DRIVE_ZIP = Path("/content/drive/MyDrive/clear_vision/datasets/merged.zip")
LOCAL_DATASET = Path("/content/datasets/merged")
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREPARED = ".prepared"  # holds merged.done's content; matching means already unpacked


def count_images(dataset: Path) -> dict[str, int]:
    return {
        split: sum(1 for p in (dataset / "images" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        for split in SPLITS
    }


def prepare_dataset(
    zip_path: Path | str = DRIVE_ZIP,
    dest: Path | str = LOCAL_DATASET,
    report_path: Path | str | None = None,
    check: bool = True,
) -> Path:
    """Unpack the verified merged.zip to `dest` and return the path of its data.yaml.

    - Fails if merged.done (written only after a verified upload) is missing.
    - Reuses an earlier unpack in this runtime if it came from the same zip.
    - Rewrites data.yaml's `path` to `dest`.
    - Compares the image count per split with merge_report.json next to the zip.
    - Runs Ultralytics' check_det_dataset (check=False skips it).
    """
    zip_path, dest = Path(zip_path), Path(dest)
    done = zip_path.with_suffix(".done")
    if not done.exists():
        raise RuntimeError(f"{done} is missing: {zip_path.name} was never verified. Run notebook 02 first.")
    info = json.loads(done.read_text())
    marker = dest / PREPARED

    if marker.exists() and json.loads(marker.read_text()) == info:
        logger.info("%s already unpacked from this zip, reusing it", dest)
    else:
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        local_zip = dest.parent / zip_path.name
        t0 = time.time()
        shutil.copyfile(zip_path, local_zip)
        if local_zip.stat().st_size != info["zip_bytes"]:
            raise RuntimeError(
                f"{zip_path} is {local_zip.stat().st_size} bytes, but merged.done says {info['zip_bytes']}"
            )
        logger.info("Copied %s (%.2f GB) in %.0fs", zip_path.name, info["zip_bytes"] / 1e9, time.time() - t0)
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(dest)
        local_zip.unlink()

    data_yaml = dest / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    data["path"] = str(dest.absolute())
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    report_path = Path(report_path) if report_path else zip_path.with_name("merge_report.json")
    if not report_path.exists():
        report_path = dest / "merge_report.json"
    expected = json.loads(report_path.read_text())["images"]
    found = count_images(dest)
    if found != {s: expected.get(s, 0) for s in SPLITS}:
        raise RuntimeError(f"Image count mismatch in {dest}: found {found}, {report_path.name} says {expected}")
    logger.info("Images per split: %s (matches %s)", found, report_path.name)

    if check:
        from ultralytics.data.utils import check_det_dataset

        check_det_dataset(str(data_yaml))
    marker.write_text(json.dumps(info))
    return data_yaml
