"""Download helpers shared by notebooks 01 (all datasets) and 02 (new sources before append).

Every download goes to a staging folder on the Colab disk first. It is then moved to
`<raw>/<name>/` and marked with a `.done` file, written last. A folder with `.done` is
skipped; a folder without it is a partial download and is replaced.

The Roboflow dataset list lives in configs/datasets.yaml.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

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


# Words that suggest a dropped source class is one we want (a mapping that missed a spelling).
WANTED_HINTS = ("curb", "kerb", "stair", "step", "pole", "pothole", "hole", "manhole", "bollard", "cone",
                "barrier", "fence", "crosswalk", "zebra", "person", "people", "pedestrian", "light")


def class_table(names: Sequence[str], mapping: dict[str, Any]) -> list[tuple[str, str]]:
    """(source class, target class or "drop") for each real class of a dataset."""
    cmap = {str(k): str(v) for k, v in (mapping.get("map") or {}).items()}
    default = "drop" if mapping.get("unlisted") == "drop" else "UNMAPPED"
    return [(n, cmap.get(n, default)) for n in names]


def suspicious_drops(table: Sequence[tuple[str, str]]) -> list[str]:
    """Dropped classes whose name looks like one of ours, e.g. "Curbs" when only "Curb" is mapped."""
    return [n for n, t in table if t == "drop" and any(h in n.lower() for h in WANTED_HINTS)]


def check_mappings(raw: Path, names: Sequence[str], load_mapping: MappingLoader) -> None:
    """Log, per downloaded dataset, its real class names, where each goes, and problems:
    unmapped classes, mapped names it lacks, dropped names that look like ours."""
    logger.info("\nClass check against configs/mappings/:")
    for name in names:
        classes = read_class_names(raw / name)
        if classes is None:
            continue
        mapping = load_mapping(name)
        if mapping is None:
            logger.warning("[%s] no mapping file configs/mappings/%s.yaml. Classes: %s", name, name, classes)
            continue
        table = class_table(classes, mapping)
        logger.info("[%s] %d real classes:", name, len(classes))
        for src, target in table:
            logger.info("    %-32s -> %s", src, target)
        problems = mapping_problems(classes, mapping)
        if mapping.get("missing") == "warn":  # guessed spellings: missing ones are expected
            problems = [p for p in problems if not p.startswith("mapping lists")]
        for problem in problems:
            logger.warning("[%s] %s", name, problem)
        for cls in suspicious_drops(table):
            logger.warning("[%s] CHECK: %r is dropped but looks like one of our classes", name, cls)
        if not problems:
            logger.info("[%s] all %d classes are mapped", name, len(classes))


# ---------------------------------------------------------------------- negatives (indoor, no labels)

# COCO classes that mark an indoor scene.
INDOOR_CLASSES = ("chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "refrigerator", "oven",
                  "microwave", "sink", "potted plant", "book", "clock", "keyboard")
# COCO classes that are (or map to) our 20 classes: an image with any of them is no negative,
# since its objects would be unlabeled. Outdoor scenes are excluded too.
FORBIDDEN_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "bench", "fire hydrant",
                     "traffic light", "stop sign", "parking meter", "train", "airplane", "boat")
NEGATIVES = "negatives"


def pick_negatives(images: Iterable[tuple[str, Iterable[str]]], n: int, seed: int = 51) -> list[str]:
    """Image ids with an indoor class and no forbidden class, `n` of them at random (sorted).

    images: (image id, COCO class names in it)."""
    indoor, forbidden = set(INDOOR_CLASSES), set(FORBIDDEN_CLASSES)
    ok = sorted(img for img, labels in images if indoor & set(labels) and not forbidden & set(labels))
    return sorted(random.Random(seed).sample(ok, min(n, len(ok))))


def write_negatives(images: Sequence[Path], dest: Path) -> int:
    """YOLO folder of background images: images/train/<name>, an EMPTY labels/train/<stem>.txt
    each (so the merge sees "no boxes", not "missing label"), and data.yaml with one dummy
    class "background" (negatives.yaml maps it to drop). Returns the image count."""
    (dest / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dest / "labels" / "train").mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copyfile(image, dest / "images" / "train" / image.name)
        (dest / "labels" / "train" / f"{image.stem}.txt").write_text("")
    (dest / "data.yaml").write_text(yaml.safe_dump({"names": ["background"]}))
    return len(images)


# Detections that disqualify a negative: budget_v1 sees one of our hazards in it, so it may
# really contain one (unlabeled stairs or a curb would teach "ignore it"), or it is exactly
# the false alarm to learn from, but then it needs a check, not a "nothing here" label.
NEGATIVE_HAZARDS = ("stairs", "curb", "pole", "pothole", "manhole", "bollard", "barrier", "traffic_cone")
SCREEN_CONF = 0.25
SCREEN_IMGSZ = 512
COCO_TRAIN_IMAGES = 118_287

# (class name, confidence, x1, y1, x2, y2), boxes normalized to the image.
ScreenDetection = tuple[str, float, float, float, float, float]
ScreenFn = Callable[[Sequence[Path]], "dict[Path, list[ScreenDetection]]"]


def hazard_hits(detections: Sequence[ScreenDetection], conf: float = SCREEN_CONF,
                hazards: Sequence[str] = NEGATIVE_HAZARDS) -> list[ScreenDetection]:
    """The detections that disqualify a negative: a hazard class at `conf` or higher."""
    return [d for d in detections if d[0] in hazards and d[1] >= conf]


def collect_negatives(
    batches: Iterable[Sequence[tuple[str, Iterable[str]]]], n: int, screen: ScreenFn | None,
    conf: float = SCREEN_CONF, seed: int = 51,
) -> tuple[list[Path], dict[Path, list[ScreenDetection]], bool]:
    """Pick indoor images without our classes (pick_negatives rules) from growing candidate
    batches, screen each new one with `screen` (a detector; None = no screening) and drop
    those with a hazard detection, until `n` remain or the batches run out.

    batches: lists of (image path, COCO class names); later batches may repeat earlier
    images (they are screened once). Returns (kept: n at most, sorted; dropped: path ->
    its hazard detections; enough: True if n were found)."""
    seen: set[str] = set()
    clean: list[Path] = []
    dropped: dict[Path, list[ScreenDetection]] = {}
    for batch in batches:
        eligible = pick_negatives(batch, COCO_TRAIN_IMAGES, seed)  # every eligible image, sorted
        new = [p for p in eligible if p not in seen]
        seen.update(new)
        paths = [Path(p) for p in new]
        found = screen(paths) if screen and paths else {}
        for path in paths:
            hits = hazard_hits(found.get(path, []), conf)
            if hits:
                dropped[path] = hits
            else:
                clean.append(path)
        logger.info("  negatives: %d candidates screened, %d clean, %d dropped", len(seen), len(clean), len(dropped))
        if len(clean) >= n:
            break
    kept = sorted(random.Random(seed).sample(clean, n)) if len(clean) > n else sorted(clean)
    return kept, dropped, len(clean) >= n


def make_yolo_screen(model_path: Path | str, imgsz: int = SCREEN_IMGSZ, conf: float = SCREEN_CONF,
                     batch: int = 32) -> ScreenFn:
    """A screen for collect_negatives(): runs an Ultralytics model (e.g. budget_v1) on the images."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names

    def screen(paths: Sequence[Path]) -> dict[Path, list[ScreenDetection]]:
        out: dict[Path, list[ScreenDetection]] = {}
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            for path, res in zip(chunk, model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, verbose=False)):
                b = res.boxes
                out[path] = [(names[int(c)], float(s), *map(float, box))
                             for box, s, c in zip(b.xyxyn.tolist(), b.conf.tolist(), b.cls.tolist())]
        return out

    return screen


def draw_hits(path: Path, hits: Sequence[ScreenDetection]) -> Any:
    """RGB image with the disqualifying detections drawn in red (for the notebook)."""
    import cv2

    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    for cls, score, x1, y1, x2, y2 in hits:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(image, p1, p2, (0, 0, 255), max(2, w // 300))
        cv2.putText(image, f"{cls} {score:.2f}", (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, w / 1200), (0, 0, 255), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def download_negatives(
    raw: Path, n: int = 3000, staging: Path = STAGING, seed: int = 51, screen: ScreenFn | None = None,
    screen_name: str = "", conf: float = SCREEN_CONF,
) -> dict[str, Any]:
    """~n indoor COCO-2017 train images without our classes into raw/negatives/ (FiftyOne zoo;
    only the needed images are downloaded), screened with `screen` (budget_v1 in notebook 02):
    images where it detects a NEGATIVE_HAZARDS class at `conf` are dropped and replaced by
    more COCO images (4x, 8x, 16x ... n candidates) until n remain or COCO has no more.

    Skipped if already done with the same screen. Returns {"kept", "dropped" (path -> hits),
    "enough"}; empty lists when skipped."""
    dest = raw / NEGATIVES
    if is_done(dest) and json.loads((dest / DONE).read_text()).get("screened_with") == screen_name:
        logger.info("[%s] already downloaded, skipping", dest.name)
        return {"kept": [], "dropped": {}, "enough": True}
    import fiftyone.zoo as foz

    t0 = time.time()

    def batches() -> Iterator[list[tuple[str, list[str]]]]:
        size, last = n * 4, -1
        while size <= COCO_TRAIN_IMAGES * 2:
            ds = foz.load_zoo_dataset(
                "coco-2017", split="train", label_types=["detections"], classes=list(INDOOR_CLASSES),
                max_samples=size, shuffle=True, seed=seed, dataset_name="coco-negatives",
                drop_existing_dataset=True,
            )
            if len(ds) == last:  # COCO has no more indoor images
                return
            last = len(ds)
            yield [(s.filepath, [d.label for d in (s.ground_truth.detections if s.ground_truth else [])]) for s in ds]
            size *= 2

    kept, dropped, enough = collect_negatives(batches(), n, screen, conf, seed)
    if not enough:
        logger.warning("  only %d negatives left after screening (wanted %d): COCO has no more indoor "
                       "images without our classes", len(kept), n)
    local = fresh_staging(staging, NEGATIVES)
    count = write_negatives(kept, local)
    publish(local, dest, {
        "source": "coco-2017", "kind": "negatives", "images": count, "wanted": n,
        "indoor_classes": list(INDOOR_CLASSES), "forbidden_classes": list(FORBIDDEN_CLASSES),
        "screened_with": screen_name, "screen_conf": conf, "screen_hazards": list(NEGATIVE_HAZARDS),
        "dropped_by_screen": len(dropped),
    })
    logger.info("  %d negatives (%d dropped by the screen) in %.0fs", count, len(dropped), time.time() - t0)
    return {"kept": kept, "dropped": dropped, "enough": enough}
