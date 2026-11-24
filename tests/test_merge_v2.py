"""v2 append features of scripts/merge_datasets.py: focus only, the pHash near-duplicate
guard, the external_test split, and the new mapping files.

The guard needs real photos (the tiny fixtures in tests/assets/merge all share one pHash),
so these tests build sources from tests/assets/street_photos.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import scripts.merge_datasets as md
from scripts.merge_datasets import EXTERNAL, Record, SourceStats

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = sorted((ROOT / "tests" / "assets" / "street_photos").glob("*.jpg"))
CLASSES = ["person", "curb", "pole", "car"]  # curb and pole are focus classes
CURB, POLE, CAR = 1, 2, 3


def jpeg(path: Path, max_side: int = 400, crop: float = 0.0, quality: int = 85) -> bytes:
    """A photo resized (and optionally cropped by `crop` of each side), as JPEG bytes."""
    image = cv2.imread(str(path))
    if crop:
        h, w = image.shape[:2]
        image = image[int(h * crop):int(h * (1 - crop)), int(w * crop):int(w * (1 - crop))]
    h, w = image.shape[:2]
    s = max_side / max(h, w)
    image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def rec(source: str, name: str, boxes: list[int], group: str | None = None, phash: int = 0,
        force: str = "") -> Record:
    r = Record(source, Path(f"{name}.jpg"), group or f"{source}/{name}", own_test=False,
               boxes=[(c, 0.5, 0.5, 0.2, 0.2) for c in boxes], phash=phash, force_split=force)
    r.out_stem = f"{source}_{name}"
    return r


def stats_for(*sources: str) -> dict[str, SourceStats]:
    return {s: SourceStats(folder=s) for s in sources}


# ---------------------------------------------------------------------- focus only

def test_focus_filter_keeps_focus_and_at_most_20_percent_others() -> None:
    records = [rec("s", f"f{i}", [CURB]) for i in range(8)] + [rec("s", f"o{i}", [CAR]) for i in range(10)]
    records += [rec("s", "bg", [])]
    kept, dropped = md.focus_filter(records, {CURB, POLE}, other_share=0.2, seed=1)
    names = {r.image.stem for r in kept}
    assert {f"f{i}" for i in range(8)} <= names  # every focus image
    others = [n for n in names if not n.startswith("f")]
    assert len(others) == 2  # 8 focus -> at most 2 others (2 / 10 = 20% of the kept images)
    assert dropped == len(records) - len(kept) == 9
    assert [r.image.stem for r in kept] == [r.image.stem for r in records if r.image.stem in names]  # order kept
    assert md.focus_filter(records, {CURB, POLE}, 0.2, seed=1)[0] == kept  # deterministic


def test_focus_filter_keeps_augmented_copies_together() -> None:
    # One group: an augmented copy whose crop lost the curb box still goes with its original.
    records = [rec("s", "a.rf.1", [CURB], group="s/a"), rec("s", "a.rf.2", [], group="s/a")]
    records += [rec("s", f"o{i}", [CAR], group=f"s/o{i}") for i in range(5)]
    kept, _ = md.focus_filter(records, {CURB}, other_share=0.0)
    assert [r.image.stem for r in kept] == ["a.rf.1", "a.rf.2"]


def test_focus_filter_never_splits_another_group() -> None:
    records = [rec("s", "f", [POLE])] + [rec("s", f"b{i}", [CAR], group="s/big") for i in range(3)]
    kept, _ = md.focus_filter(records, {POLE}, other_share=0.5)  # budget 1 image: the 3-image group doesn't fit
    assert [r.image.stem for r in kept] == ["f"]


def test_apply_focus_only_filters_only_listed_sources() -> None:
    records = [rec("a", "f", [CURB]), rec("a", "o1", [CAR]), rec("a", "o2", [CAR]),
               rec("b", "o1", [CAR]), rec("b", "o2", [CAR])]
    stats = stats_for("a", "b")
    kept = md.apply_focus_only(records, ["a"], CLASSES, stats, other_share=0.2)
    assert [(r.source, r.image.stem) for r in kept] == [("a", "f"), ("b", "o1"), ("b", "o2")]
    assert stats["a"].focus_dropped == 2 and stats["b"].focus_dropped == 0


def test_focus_classes_are_the_five_weakest_hazards() -> None:
    assert md.FOCUS_CLASSES == ("curb", "stairs", "pothole", "pole", "manhole")
    assert md.OTHER_SHARE == 0.2 and md.PHASH_DISTANCE == 6


# ---------------------------------------------------------------------- pHash and distances

def test_phash_finds_resized_and_recompressed_copies_but_not_other_photos() -> None:
    a = md.phash_of(jpeg(PHOTOS[0], 800, quality=90))
    copy = md.phash_of(jpeg(PHOTOS[0], 320, quality=60))  # another export of the same photo
    others = [md.phash_of(jpeg(p, 800)) for p in PHOTOS[1:]]
    assert bin(a ^ copy).count("1") <= md.PHASH_DISTANCE
    assert min(bin(a ^ o).count("1") for o in others) > md.PHASH_DISTANCE


def test_near_matches_and_has_near() -> None:
    a = [0b0000, 0b1111, 1 << 63]
    b = [0b0001, (1 << 63) | 0b111]
    assert md.near_matches(a, b, 1) == [(0, 0)]
    assert sorted(md.near_matches(a, b, 2)) == [(0, 0), (1, 1), (2, 0)]
    assert sorted(md.near_matches(a, b, 3)) == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert md.has_near(a, b, 1) == [True, False, False]
    assert md.near_matches(a, b, -1) == []  # negative distance: guard off
    assert md.near_matches([], b, 6) == [] and md.near_matches(a, [], 6) == []


def test_popcount_fallback_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.array([0, 1, 2**63 + 5, 2**64 - 1], np.uint64)
    fast = md._popcount(x)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert list(md._popcount(x)) == list(fast) == [0, 1, 3, 64]


# ---------------------------------------------------------------------- guard against merged images

def test_guard_against_merged_drops_held_out_and_moves_train() -> None:
    merged = {"train/t.jpg": 0b0, "val/v.jpg": 0xFF00, "test/x.jpg": 0xF0F0F0F0, f"{EXTERNAL}/e.jpg": 2**40 - 1}
    records = [
        rec("n", "near_train", [CAR], phash=0b11),  # 2 bits from train
        rec("n", "near_val", [CAR], phash=0xFF01),
        rec("n", "near_test", [CAR], phash=0xF0F0F0F1),
        rec("n", "near_external", [CAR], phash=2**40 - 2),
        rec("n", "fresh", [CAR], phash=0x123456789ABCDEF0),
        rec("x", "external_near_train", [CAR], phash=0b1, force=EXTERNAL),
    ]
    stats = stats_for("n", "x")
    kept = md.guard_against_merged(records, merged, stats)
    assert [(r.image.stem, r.force_split) for r in kept] == [("near_train", "train"), ("fresh", "")]
    assert stats["n"].near_dup_dropped == 3 and stats["n"].near_dup_to_train == 1
    assert stats["x"].near_dup_dropped == 1  # the external test stays unseen


# ---------------------------------------------------------------------- guard among new images

def test_guard_new_records_joins_same_source_near_duplicates_into_one_group() -> None:
    records = [rec("n", "a", [CAR], phash=0), rec("n", "b", [CAR], phash=1),
               rec("n", "c", [CAR], phash=0xFFFF_FFFF_FFFF_0000)]
    kept = md.guard_new_records(records, stats_for("n"))
    assert kept[0].group == kept[1].group != kept[2].group
    md.assign_splits(kept, (0.4, 0.3, 0.3), seed=3)
    assert kept[0].split == kept[1].split


def test_guard_new_records_puts_cross_source_near_duplicates_in_train() -> None:
    records = [rec("a", "x", [CAR], phash=0), rec("b", "y", [CAR], phash=3)]
    stats = stats_for("a", "b")
    kept = md.guard_new_records(records, stats)
    assert [r.force_split for r in kept] == ["train", "train"]
    md.assign_splits(kept, (0.0, 0.5, 0.5), seed=1)  # no train share at all: still train
    assert [r.split for r in kept] == ["train", "train"]
    assert stats["a"].near_dup_to_train == stats["b"].near_dup_to_train == 1


def test_guard_new_records_spreads_forced_train_over_its_cluster() -> None:
    records = [rec("a", "x", [CAR], phash=0, force="train"), rec("a", "y", [CAR], phash=1)]
    kept = md.guard_new_records(records, stats_for("a"))
    assert [r.force_split for r in kept] == ["train", "train"]


def test_guard_new_records_drops_external_near_other_new_images() -> None:
    records = [rec("train_src", "x", [CAR], phash=0), rec("ext", "y", [CAR], phash=1, force=EXTERNAL),
               rec("ext", "z", [CAR], phash=2, force=EXTERNAL)]
    stats = stats_for("train_src", "ext")
    kept = md.guard_new_records(records, stats)
    assert [r.image.stem for r in kept] == ["x"]
    assert stats["ext"].near_dup_dropped == 2 and kept[0].force_split == ""


def test_assign_splits_honours_forced_splits() -> None:
    records = [rec("e", f"e{i}", [CAR], force=EXTERNAL) for i in range(3)]
    records += [rec("s", f"t{i}", [CAR], force="train") for i in range(3)]
    md.assign_splits(records, (0.0, 0.0, 1.0), seed=1)
    assert [r.split for r in records] == [EXTERNAL] * 3 + ["train"] * 3


# ---------------------------------------------------------------------- end to end on real photos

def write_source(root: Path, name: str, items: list[tuple[str, bytes, str]], split: str = "train",
                 names: list[str] | None = None) -> None:
    """A Roboflow-style YOLO source: <split>/images/<file>, <split>/labels/<stem>.txt."""
    for sub in ("images", "labels"):
        (root / name / split / sub).mkdir(parents=True, exist_ok=True)
    for stem, data, label in items:
        (root / name / split / "images" / f"{stem}.jpg").write_bytes(data)
        (root / name / split / "labels" / f"{stem}.txt").write_text(label)
    (root / name / "data.yaml").write_text(yaml.safe_dump({"names": names or ["Curb", "Pole", "car"]}))


def write_mapping(mappings: Path, name: str, extra: str = "") -> None:
    mappings.mkdir(parents=True, exist_ok=True)
    (mappings / f"{name}.yaml").write_text(
        f"folder: {name}\nunlisted: drop\nmissing: warn\n{extra}map:\n  Curb: curb\n  Pole: pole\n  car: car\n"
    )


CURB_BOX, POLE_BOX, CAR_BOX = "0 0.5 0.5 0.3 0.3\n", "1 0.5 0.5 0.1 0.6\n", "2 0.5 0.5 0.3 0.2\n"


def test_append_focus_only_guard_and_external_split(tmp_path: Path) -> None:
    raw, mappings = tmp_path / "raw", tmp_path / "mappings"
    (tmp_path / "classes.yaml").write_text(yaml.safe_dump({"names": CLASSES}))
    zip_path = tmp_path / "drive" / "merged.zip"
    p = iter(PHOTOS)

    # Merged dataset: 3 train photos and 2 photos in the source's own test split.
    base_train = [next(p) for _ in range(3)]
    base_test = [next(p) for _ in range(2)]
    write_source(raw, "base", [(f"t{i}", jpeg(x), CURB_BOX) for i, x in enumerate(base_train)])
    write_source(raw, "base", [(f"x{i}", jpeg(x), CURB_BOX) for i, x in enumerate(base_test)], split="test")
    write_mapping(mappings, "base")
    common = dict(work=tmp_path / "work", classes_path=tmp_path / "classes.yaml", mappings_dir=mappings,
                  ratios=(1.0, 0.0, 0.0), sync=lambda: None)
    md.merge(raw, zip_path, **common)

    # New source "fresh": other exports of a train and a test photo, 4 new focus photos,
    # 6 car-only photos (focus only keeps at most 1), and an augmented pair.
    focus_new = [next(p) for _ in range(4)]
    cars = [next(p) for _ in range(6)]
    pair = next(p)
    write_source(raw, "fresh", [
        ("copy_of_train", jpeg(base_train[0], 320, quality=60), POLE_BOX),
        ("copy_of_test", jpeg(base_test[0], 320, quality=60), POLE_BOX),
        *[(f"f{i}", jpeg(x), POLE_BOX) for i, x in enumerate(focus_new)],
        *[(f"c{i}", jpeg(x), CAR_BOX) for i, x in enumerate(cars)],
        ("pair_jpg.rf.a", jpeg(pair), CURB_BOX), ("pair_jpg.rf.b", jpeg(pair, crop=0.02), CURB_BOX),
    ])
    write_mapping(mappings, "fresh")
    # External source: one new photo, and another export of a "fresh" focus photo (dropped).
    ext_photo = next(p)
    write_source(raw, "ext", [("e0", jpeg(ext_photo), CURB_BOX), ("e_copy", jpeg(focus_new[1], 300), CURB_BOX)])
    write_mapping(mappings, "ext", extra=f"split: {EXTERNAL}\n")

    report = md.append(raw, zip_path, ["fresh", "ext"], focus_only=["fresh", "ext"],
                       ratios=(0.5, 0.25, 0.25), **{k: v for k, v in common.items() if k != "ratios"})

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        data = yaml.safe_load(zf.read("data.yaml"))
        phash_index = json.loads(zf.read(md.PHASH_FILE))
    images = {n.split("/")[-1]: n.split("/")[1] for n in names if n.startswith("images/")}

    fresh, ext = report["sources"]["fresh"], report["sources"]["ext"]
    # Focus only (before the guard): 8 focus images (2 copies, 4 photos, the pair), so at most
    # 2 car-only images (2 of 10 = 20%).
    kept_cars = [n for n in images if n.startswith("fresh_c") and n[7].isdigit()]
    assert len(kept_cars) == 2 and fresh["focus_dropped"] == 4
    # Leakage guard: the copy of a test image is gone; the copy of a train image is in train.
    assert "fresh_copy_of_test.jpg" not in images
    assert images["fresh_copy_of_train.jpg"] == "train"
    assert fresh["near_dup_dropped"] == 1 and fresh["near_dup_to_train"] >= 1
    # Augmented copies share a split.
    assert images["fresh_pair_jpg.rf.a.jpg"] == images["fresh_pair_jpg.rf.b.jpg"]
    # External: its own split, listed in data.yaml; the copy of a training photo is dropped.
    assert images["ext_e0.jpg"] == EXTERNAL and "ext_e_copy.jpg" not in images
    assert ext["near_dup_dropped"] == 1 and ext["images"] == {"train": 0, "val": 0, "test": 0, EXTERNAL: 1}
    assert data[EXTERNAL] == f"images/{EXTERNAL}"
    assert report["images"][EXTERNAL] == 1 and report["boxes"][EXTERNAL]["curb"] == 1
    # The pHash index covers every image, for the next append.
    assert set(phash_index) == {f"{split}/{name}" for name, split in images.items()}
    assert md.load_classes(tmp_path / "classes.yaml") == CLASSES


def test_append_negatives_keeps_empty_label_images(tmp_path: Path) -> None:
    from scripts import download_datasets as dl

    raw, mappings = tmp_path / "raw", tmp_path / "mappings"
    (tmp_path / "classes.yaml").write_text(yaml.safe_dump({"names": CLASSES}))
    zip_path = tmp_path / "drive" / "merged.zip"
    write_source(raw, "base", [(f"t{i}", jpeg(x), CURB_BOX) for i, x in enumerate(PHOTOS[:2])])
    write_mapping(mappings, "base")
    common = dict(work=tmp_path / "work", classes_path=tmp_path / "classes.yaml", mappings_dir=mappings,
                  sync=lambda: None)
    md.merge(raw, zip_path, ratios=(1.0, 0.0, 0.0), **common)

    files = []
    for i, photo in enumerate(PHOTOS[10:14]):
        (tmp_path / "coco").mkdir(exist_ok=True)
        (tmp_path / "coco" / f"{i:012d}.jpg").write_bytes(jpeg(photo))
        files.append(tmp_path / "coco" / f"{i:012d}.jpg")
    assert dl.write_negatives(files, raw / "negatives") == 4
    (mappings / "negatives.yaml").write_text("folder: negatives\nunlisted: drop\nmap:\n  background: drop\n")
    report = md.append(raw, zip_path, ["negatives"], ratios=(0.5, 0.25, 0.25), **common)

    neg = report["sources"]["negatives"]
    assert sum(neg["images"].values()) == 4 and neg["missing_label_files"] == 0
    assert all(not b for b in neg["boxes"].values())


# ---------------------------------------------------------------------- the real mapping files

def test_new_sources_have_mappings_and_rf_safewalk_is_external() -> None:
    classes = md.load_classes(ROOT / "configs" / "classes.yaml")
    mappings = {m.short_name: m for m in md.load_mappings(ROOT / "configs" / "mappings", classes)}
    datasets = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())["roboflow"]
    new = {"rf_freesight": ("kaan-bisfl", "freesight-akli2", 1), "rf_neytra": ("jeet-xlgwq", "neytra-v3-sxshi", 4),
           "rf_safewalk": ("safewalkbd-7jrsx", "safewalkbd-final", 31),
           "rf_drone_obstacles": ("droneobstacles-bvrmw", "drone-obstacles-acuxy", 4)}
    for name, (workspace, project, version) in new.items():
        entry = next(d for d in datasets if d["name"] == name)
        assert (entry["workspace"], entry["project"], entry["version"]) == (workspace, project, version)
        m = mappings[name]
        assert m.unlisted_drop and m.missing_warn
        assert m.classes["Curb"] == "curb" and m.classes["staircase"] == "stairs" and m.classes["Pole"] == "pole"
        assert m.classes["Road-barrier"] == "barrier" and m.classes["fence"] == "barrier"
        assert m.classes["traffic cone"] == "traffic_cone" and m.classes["People"] == "person"
        assert m.classes["two wheeler"] == "drop" and m.classes["Light"] == "drop" and "Light" in m.review
        assert m.classes["Traffic-light"] == "traffic_light"
    assert mappings["rf_safewalk"].split == EXTERNAL
    assert all(mappings[n].split == "" for n in new if n != "rf_safewalk")
    assert mappings["negatives"].classes == {"background": "drop"}


def test_bad_split_value_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text("folder: x\nsplit: holdout\nmap:\n  a: drop\n")
    with pytest.raises(md.MergeError, match="split must be"):
        md.load_mappings(tmp_path, CLASSES)
