"""Download helpers shared by notebooks 01 (all datasets) and 02 (new sources before append).

Every download goes to a staging folder on the Colab disk first. It is then moved to
`<raw>/<name>/` and marked with a `.done` file, written last. A folder with `.done` is
skipped; a folder without it is a partial download and is replaced.

The Roboflow dataset list lives in configs/datasets.yaml.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

logger = logging.getLogger("download")

REPO = Path(__file__).resolve().parent.parent
DONE = ".done"
STAGING = Path("/content/staging")  # Colab disk: much faster than writing file by file to Drive
ROBOFLOW_FORMAT = "yolov8"

MappingLoader = Callable[[str], "dict[str, Any] | None"]


@dataclass(frozen=True)
class RoboflowDataset:
    name: str  # folder name under raw/, and the mapping file name
    workspace: str
    project: str
    version: int


# ---------------------------------------------------------------------- folders


def is_done(dest: Path) -> bool:
    return (dest / DONE).exists()


def skip_if_done(dest: Path) -> bool:
    if is_done(dest):
        logger.info("[%s] already downloaded, skipping", dest.name)
        return True
    if dest.exists():
        logger.info("[%s] partial download found, it will be replaced", dest.name)
    return False


def fresh_staging(staging: Path, name: str) -> Path:
    local = staging / name
    shutil.rmtree(local, ignore_errors=True)
    return local


def publish(local: Path, dest: Path, info: dict[str, Any]) -> None:
    """Move a finished download from staging to `dest`, then mark it done."""
    t0 = time.time()
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(local), str(dest))
    (dest / DONE).write_text(json.dumps(info, indent=2))
    logger.info("  saved to %s in %.0fs", dest, time.time() - t0)


def record_failure(failures: dict[str, str], name: str, error: Exception) -> None:
    failures[name] = f"{type(error).__name__}: {error}"
    logger.error("  FAILED: %s", failures[name])


def colab_secret(name: str) -> str | None:
    """A Colab Secret, or None (with instructions) if it is missing or not shared."""
    try:
        from google.colab import userdata

        value = userdata.get(name)
        logger.info("%s loaded", name)
        return value
    except Exception as e:
        logger.warning(
            "WARNING: %s not available (%s). Add it under Secrets (key icon) with notebook "
            "access, then re-run.", name, type(e).__name__,
        )
        return None


# ---------------------------------------------------------------------- Roboflow


def load_roboflow_datasets(path: Path = REPO / "configs" / "datasets.yaml") -> list[RoboflowDataset]:
    entries = (yaml.safe_load(Path(path).read_text()) or {}).get("roboflow") or []
    datasets = [
        RoboflowDataset(str(e["name"]), str(e["workspace"]), str(e["project"]), int(e["version"]))
        for e in entries
    ]
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate dataset names in {names}")
    return datasets


def select_datasets(datasets: Sequence[RoboflowDataset], names: Sequence[str]) -> list[RoboflowDataset]:
    """The datasets called `names` (all if empty). Unknown names are an error."""
    if not names:
        return list(datasets)
    known = {d.name for d in datasets}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}. Known: {sorted(known)} (configs/datasets.yaml)")
    return [d for d in datasets if d.name in names]


def version_number(version: Any) -> int:
    return int(str(version.version).split("/")[-1])


def download_roboflow(rf: Any, ds: RoboflowDataset, local: Path) -> int:
    """Download one Roboflow version, or the latest one if it fails. Returns the version used."""
    project = rf.workspace(ds.workspace).project(ds.project)
    try:
        project.version(ds.version).download(ROBOFLOW_FORMAT, location=str(local), overwrite=True)
        return ds.version
    except Exception as e:
        logger.warning("  version %d failed (%s: %s), trying the latest version", ds.version, type(e).__name__, e)
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"{ds.workspace}/{ds.project} has no versions")
    latest = max(version_number(v) for v in versions)
    shutil.rmtree(local, ignore_errors=True)
    project.version(latest).download(ROBOFLOW_FORMAT, location=str(local), overwrite=True)
    logger.warning("  USED LATEST VERSION %d instead of %d", latest, ds.version)
    return latest


def download_roboflow_datasets(
    datasets: Sequence[RoboflowDataset],
    raw: Path,
    api_key: str | None,
    staging: Path = STAGING,
    rf: Any = None,
) -> dict[str, str]:
    """Download each dataset into `raw/<name>/`, skipping finished ones.
    Returns {name: error} for the ones that failed. `rf` is for tests."""
    failures: dict[str, str] = {}
    raw.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        dest = raw / ds.name
        if skip_if_done(dest):
            continue
        logger.info("\n[%s] %s/%s v%d", ds.name, ds.workspace, ds.project, ds.version)
        if rf is None:
            if not api_key:
                record_failure(failures, ds.name, RuntimeError("no ROBOFLOW_API_KEY"))
                continue
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
        local = fresh_staging(staging, ds.name)
        t0 = time.time()
        try:
            used = download_roboflow(rf, ds, local)
            logger.info("  downloaded v%d in %.0fs", used, time.time() - t0)
            publish(local, dest, {
                "source": "roboflow", "workspace": ds.workspace, "project": ds.project,
                "requested_version": ds.version, "version": used, "format": ROBOFLOW_FORMAT,
            })
        except Exception as e:
            record_failure(failures, ds.name, e)
    return failures


# ---------------------------------------------------------------------- class check


def read_class_names(dataset_dir: Path) -> list[str] | None:
    """Class names from a dataset's data.yaml / dataset.yaml, or None if it has neither."""
    for name in ("data.yaml", "dataset.yaml"):
        path = dataset_dir / name
        if path.exists():
            names = (yaml.safe_load(path.read_text()) or {}).get("names") or []
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=int)]
            return [str(n) for n in names]
    return None


def mapping_problems(names: Sequence[str], mapping: dict[str, Any]) -> list[str]:
    """Messages about classes missing from a mapping file, and mapped classes not in `names`."""
    cmap = {str(k): str(v) for k, v in (mapping.get("map") or {}).items()}
    unmapped = [n for n in names if n not in cmap]
    missing = [c for c in cmap if c not in names]
    problems = []
    if unmapped and mapping.get("unlisted") == "drop":
        problems.append(f"{len(unmapped)} unmapped classes, dropped by `unlisted: drop`: {unmapped}")
    elif unmapped:
        problems.append(f"UNMAPPED classes {unmapped}: the merge fails until they are in the mapping file")
    if missing:
        problems.append(f"mapping lists classes this dataset does not have: {missing}")
    return problems


def mappings_from_dir(mappings_dir: Path) -> MappingLoader:
    def load(name: str) -> dict[str, Any] | None:
        path = Path(mappings_dir) / f"{name}.yaml"
        return (yaml.safe_load(path.read_text()) or {}) if path.exists() else None

    return load


def check_mappings(raw: Path, names: Sequence[str], load_mapping: MappingLoader) -> None:
    """Log, per downloaded dataset, the classes its mapping file doesn't cover."""
    logger.info("\nClass check against configs/mappings/:")
    for name in names:
        classes = read_class_names(raw / name)
        if classes is None:
            continue
        mapping = load_mapping(name)
        if mapping is None:
            logger.warning("[%s] no mapping file configs/mappings/%s.yaml. Classes: %s", name, name, classes)
            continue
        problems = mapping_problems(classes, mapping)
        for problem in problems:
            logger.warning("[%s] %s", name, problem)
        if not problems:
            logger.info("[%s] all %d classes are mapped", name, len(classes))
