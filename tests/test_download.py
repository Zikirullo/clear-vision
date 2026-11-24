"""Tests for scripts/download_datasets.py (shared by notebooks 01 and 02), with a fake Roboflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import download_datasets as dl
from scripts.download_datasets import (
    DONE,
    REPO,
    RoboflowDataset,
    check_mappings,
    download_roboflow_datasets,
    load_roboflow_datasets,
    mapping_problems,
    mappings_from_dir,
    read_class_names,
    select_datasets,
)

NOTEBOOKS = REPO / "notebooks"


class FakeProject:
    def __init__(self, available: list[int], broken: tuple[int, ...] = (), names: tuple[str, ...] = ("cone",)) -> None:
        self.available, self.broken, self.names = available, broken, list(names)
        self.calls: list[int] = []

    def version(self, number: int) -> SimpleNamespace:
        def download(fmt: str, location: str, overwrite: bool) -> None:
            self.calls.append(number)
            if number in self.broken or number not in self.available:
                raise RuntimeError(f"version {number} not found")
            (Path(location) / "train" / "images").mkdir(parents=True)
            (Path(location) / "data.yaml").write_text(f"names: {self.names}\n")

        return SimpleNamespace(download=download)

    def versions(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(version=f"ws/project/{n}") for n in self.available]


class FakeRoboflow:
    def __init__(self, projects: dict[str, FakeProject]) -> None:
        self.projects = projects

    def workspace(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(project=lambda project: self.projects[project])


def ds(name: str, version: int = 1) -> RoboflowDataset:
    return RoboflowDataset(name=name, workspace="ws", project=name, version=version)


def download(tmp_path: Path, datasets: list[RoboflowDataset], rf: FakeRoboflow | None, api_key: str | None = "key") -> dict:
    return download_roboflow_datasets(datasets, tmp_path / "raw", api_key, staging=tmp_path / "staging", rf=rf)


# ---------------------------------------------------------------------- downloads


def test_downloads_into_raw_and_marks_done(tmp_path: Path) -> None:
    rf = FakeRoboflow({"rf_cone": FakeProject([1, 2])})
    assert download(tmp_path, [ds("rf_cone", 2)], rf) == {}
    dest = tmp_path / "raw" / "rf_cone"
    assert (dest / "data.yaml").exists()
    info = json.loads((dest / DONE).read_text())
    assert info["version"] == info["requested_version"] == 2
    assert not (tmp_path / "staging" / "rf_cone").exists()


def test_skips_finished_download(tmp_path: Path) -> None:
    project = FakeProject([1])
    dest = tmp_path / "raw" / "rf_cone"
    dest.mkdir(parents=True)
    (dest / DONE).write_text("{}")
    assert download(tmp_path, [ds("rf_cone")], FakeRoboflow({"rf_cone": project})) == {}
    assert project.calls == []


def test_replaces_partial_download(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "rf_cone"
    dest.mkdir(parents=True)
    (dest / "leftover.txt").write_text("partial")
    download(tmp_path, [ds("rf_cone")], FakeRoboflow({"rf_cone": FakeProject([1])}))
    assert (dest / DONE).exists()
    assert not (dest / "leftover.txt").exists()


def test_falls_back_to_latest_version(tmp_path: Path) -> None:
    project = FakeProject([1, 3, 2])
    assert download(tmp_path, [ds("rf_cone", 6)], FakeRoboflow({"rf_cone": project})) == {}
    assert project.calls == [6, 3]
    assert json.loads((tmp_path / "raw" / "rf_cone" / DONE).read_text())["version"] == 3


def test_failure_is_recorded_and_others_continue(tmp_path: Path) -> None:
    rf = FakeRoboflow({"rf_bad": FakeProject([]), "rf_cone": FakeProject([1])})
    failures = download(tmp_path, [ds("rf_bad"), ds("rf_cone")], rf)
    assert list(failures) == ["rf_bad"]
    assert "no versions" in failures["rf_bad"]
    assert not (tmp_path / "raw" / "rf_bad").exists()
    assert (tmp_path / "raw" / "rf_cone" / DONE).exists()


def test_missing_api_key_fails_each_dataset(tmp_path: Path) -> None:
    failures = download(tmp_path, [ds("rf_a"), ds("rf_b")], rf=None, api_key=None)
    assert failures == {"rf_a": "RuntimeError: no ROBOFLOW_API_KEY", "rf_b": "RuntimeError: no ROBOFLOW_API_KEY"}


# ---------------------------------------------------------------------- dataset list


def test_select_datasets() -> None:
    datasets = [ds("rf_a"), ds("rf_b"), ds("rf_c")]
    assert select_datasets(datasets, []) == datasets
    assert [d.name for d in select_datasets(datasets, ["rf_c", "rf_a"])] == ["rf_a", "rf_c"]
    with pytest.raises(ValueError, match="rf_typo"):
        select_datasets(datasets, ["rf_a", "rf_typo"])


def test_real_dataset_list_matches_mapping_files() -> None:
    datasets = {d.name: d for d in load_roboflow_datasets()}
    assert datasets["rf_sidewalk"] == RoboflowDataset("rf_sidewalk", "scottsdale", "sidewalk-otzhb", 1)
    assert datasets["rf_barricade"] == RoboflowDataset("rf_barricade", "myplace-65bsf", "barricade-detection-kaaty", 10)
    assert "rf_sidewalk_obstacles" not in datasets
    for name in datasets:
        assert (REPO / "configs" / "mappings" / f"{name}.yaml").exists(), name


# ---------------------------------------------------------------------- class check


def test_read_class_names(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "data.yaml").write_text("names: ['2', crosswalk]\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "dataset.yaml").write_text("names:\n  0: Person\n  1: Car\n")
    (tmp_path / "c").mkdir()
    assert read_class_names(tmp_path / "a") == ["2", "crosswalk"]
    assert read_class_names(tmp_path / "b") == ["Person", "Car"]
    assert read_class_names(tmp_path / "c") is None


def test_mapping_problems() -> None:
    mapping = {"map": {"barricade": "barrier", "fire_hydrant": "fire_hydrant"}}
    names = ["barricade", "fire hydrant"]
    assert mapping_problems(names, mapping) == [
        "UNMAPPED classes ['fire hydrant']: the merge fails until they are in the mapping file",
        "mapping lists classes this dataset does not have: ['fire_hydrant']",
    ]
    dropped = mapping_problems(names, {**mapping, "unlisted": "drop"})
    assert dropped[0] == "1 unmapped classes, dropped by `unlisted: drop`: ['fire hydrant']"
    assert mapping_problems(["barricade"], {"map": {"barricade": "barrier"}}) == []


def test_check_mappings_logs_problems(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    raw = tmp_path / "raw"
    (raw / "rf_barricade").mkdir(parents=True)
    (raw / "rf_barricade" / "data.yaml").write_text("names: [Barricade, cone]\n")
    (raw / "rf_new").mkdir()
    (raw / "rf_new" / "data.yaml").write_text("names: [x]\n")
    with caplog.at_level(logging.INFO, logger="download"):
        check_mappings(raw, ["rf_barricade", "rf_new", "rf_absent"], mappings_from_dir(REPO / "configs" / "mappings"))
    assert "[rf_barricade] UNMAPPED classes ['cone']" in caplog.text
    assert "[rf_new] no mapping file" in caplog.text
    assert "rf_absent" not in caplog.text


# ---------------------------------------------------------------------- notebooks


def notebook_code(name: str) -> list[str]:
    cells = json.loads((NOTEBOOKS / name).read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


@pytest.mark.parametrize("name", ["01_download_datasets.ipynb", "02_merge_datasets.ipynb"])
def test_notebook_code_compiles_and_reuses_shared_module(name: str) -> None:
    code = notebook_code(name)
    for i, source in enumerate(code):
        python = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        compile(python, f"{name} cell {i}", "exec")
    joined = "\n".join(code)
    assert "def download_roboflow" not in joined  # no copy of the shared download code
    assert "download_roboflow_datasets(" in joined


def test_notebook_02_downloads_before_append() -> None:
    code = notebook_code("02_merge_datasets.ipynb")
    assert 'DOWNLOAD_NEW = ["rf_freesight", "rf_neytra", "rf_safewalk", "rf_drone_obstacles"]' in code[0]
    assert '"negatives"]' in code[0] and "APPEND = True" in code[0] and "APPEND_MAX_SIDE = 800" in code[0]
    download = next(i for i, c in enumerate(code) if "download_roboflow_datasets(" in c)
    merge = next(i for i, c in enumerate(code) if "merge_datasets.append(" in c)
    assert download < merge
    assert "Path(LOCAL_RAW)" in code[download] and "download_negatives(" in code[download]
    assert "focus_only=" in code[merge] and "phash_distance=PHASH_DISTANCE" in code[merge]


# ---------------------------------------------------------------------- v2: class check, negatives

def test_class_table_and_suspicious_drops() -> None:
    mapping = {"unlisted": "drop", "map": {"Curb": "curb", "Light": "drop"}}
    table = dl.class_table(["Curb", "Curbs", "Light", "tree"], mapping)
    assert table == [("Curb", "curb"), ("Curbs", "drop"), ("Light", "drop"), ("tree", "drop")]
    assert dl.suspicious_drops(table) == ["Curbs", "Light"]  # look like ours: check the mapping
    assert dl.class_table(["x"], {"map": {}}) == [("x", "UNMAPPED")]


def test_pick_negatives_needs_indoor_and_none_of_our_classes() -> None:
    images = [
        ("office", ["chair", "laptop"]),
        ("kitchen", ["oven", "sink", "book"]),
        ("with_person", ["chair", "person"]),  # a person would be an unlabeled object
        ("street", ["car", "potted plant"]),
        ("nothing_indoor", ["kite"]),
        ("bedroom", ["bed"]),
    ]
    assert dl.pick_negatives(images, 10) == ["bedroom", "kitchen", "office"]
    two = dl.pick_negatives(images, 2, seed=1)
    assert len(two) == 2 and two == dl.pick_negatives(images, 2, seed=1)
    assert "person" in dl.FORBIDDEN_CLASSES and "traffic light" in dl.FORBIDDEN_CLASSES


def test_write_negatives_has_empty_labels(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for n in ("a.jpg", "b.jpg"):
        (src / n).write_bytes(b"jpeg")
    assert dl.write_negatives(sorted(src.iterdir()), tmp_path / "negatives") == 2
    out = tmp_path / "negatives"
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["a.jpg", "b.jpg"]
    assert [(out / "labels" / "train" / f"{s}.txt").read_text() for s in "ab"] == ["", ""]
    assert dl.read_class_names(out) == ["background"]


# ---------------------------------------------------------------------- negatives screened with budget_v1

def fake_screen(hazardous: dict[str, list]) -> tuple:
    """A detector that 'sees' the given detections in images named in `hazardous`; it records
    every image it was run on."""
    calls: list[str] = []

    def screen(paths):
        calls.extend(p.name for p in paths)
        return {p: hazardous.get(p.name, []) for p in paths}

    return screen, calls


def test_hazard_hits_only_our_hazards_at_conf() -> None:
    dets = [("stairs", 0.3, 0, 0, 1, 1), ("curb", 0.2, 0, 0, 1, 1), ("car", 0.9, 0, 0, 1, 1),
            ("traffic_cone", 0.25, 0, 0, 1, 1)]
    assert [d[0] for d in dl.hazard_hits(dets)] == ["stairs", "traffic_cone"]
    assert set(dl.NEGATIVE_HAZARDS) == {"stairs", "curb", "pole", "pothole", "manhole", "bollard", "barrier",
                                        "traffic_cone"}
    assert (dl.SCREEN_CONF, dl.SCREEN_IMGSZ) == (0.25, 512)


def test_collect_negatives_drops_hazards_and_tops_up_from_more_images() -> None:
    indoor = ["chair"]
    first = [(f"/coco/{i}.jpg", indoor) for i in range(4)] + [("/coco/person.jpg", ["chair", "person"])]
    second = first + [(f"/coco/{i}.jpg", indoor) for i in range(4, 8)]  # a bigger batch repeats the first
    screen, calls = fake_screen({
        "0.jpg": [("stairs", 0.8, 0.1, 0.1, 0.5, 0.9)],
        "2.jpg": [("pole", 0.26, 0.4, 0.0, 0.5, 1.0)],
        "3.jpg": [("pole", 0.10, 0.4, 0.0, 0.5, 1.0), ("car", 0.9, 0, 0, 1, 1)],  # weak pole, car: kept
        "5.jpg": [("curb", 0.5, 0.0, 0.8, 1.0, 1.0)],
    })
    kept, dropped, enough = dl.collect_negatives([first, second, AssertionError], 4, screen)
    assert enough and len(kept) == 4  # stopped before the third batch
    assert {p.name for p in dropped} == {"0.jpg", "2.jpg", "5.jpg"}  # 2 clean after batch 1 -> batch 2
    assert not {p.name for p in kept} & {"0.jpg", "2.jpg", "5.jpg", "person.jpg"}
    assert sorted(calls) == sorted(set(calls))  # each image screened once
    assert "person.jpg" not in calls  # not eligible: never screened
    assert dropped[Path("/coco/0.jpg")] == [("stairs", 0.8, 0.1, 0.1, 0.5, 0.9)]


def test_collect_negatives_says_when_there_are_not_enough() -> None:
    batch = [(f"/coco/{i}.jpg", ["sink"]) for i in range(3)]
    screen, _ = fake_screen({"1.jpg": [("manhole", 0.9, 0, 0, 1, 1)]})
    kept, dropped, enough = dl.collect_negatives([batch, batch], 5, screen)
    assert not enough and [p.name for p in kept] == ["0.jpg", "2.jpg"] and len(dropped) == 1


def test_collect_negatives_without_screen_keeps_all_eligible() -> None:
    batch = [(f"/coco/{i}.jpg", ["bed"]) for i in range(5)]
    kept, dropped, enough = dl.collect_negatives([batch], 3, None, seed=7)
    assert enough and len(kept) == 3 and not dropped
    assert kept == dl.collect_negatives([batch], 3, None, seed=7)[0]


def test_download_negatives_screens_tops_up_and_skips_when_done(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import ModuleType

    import cv2
    import numpy as np

    coco = tmp_path / "coco"
    coco.mkdir()
    for i in range(12):
        cv2.imwrite(str(coco / f"{i:03d}.jpg"), np.full((20, 20, 3), i * 20, np.uint8))
    sizes: list[int] = []

    def load_zoo_dataset(name, split, label_types, classes, max_samples, **kwargs):
        sizes.append(max_samples)
        det = SimpleNamespace
        return [det(filepath=str(coco / f"{i:03d}.jpg"), ground_truth=det(detections=[det(label="tv")]))
                for i in range(min(max_samples, 12))]

    zoo = ModuleType("fiftyone.zoo")
    zoo.load_zoo_dataset = load_zoo_dataset
    monkeypatch.setitem(sys.modules, "fiftyone", ModuleType("fiftyone"))
    monkeypatch.setitem(sys.modules, "fiftyone.zoo", zoo)
    # 4 candidates per wanted image -> 8 first; 3 of them show hazards -> top up to 12.
    screen, _ = fake_screen({f"{i:03d}.jpg": [("bollard", 0.7, 0, 0, 1, 1)] for i in (1, 4, 6)})
    raw = tmp_path / "raw"
    result = dl.download_negatives(raw, n=2 * 3, staging=tmp_path / "staging", screen=screen, screen_name="budget_v1")
    assert sizes == [24]  # 6 wanted -> 24 candidates asked; the fake has 12, enough after screening
    assert result["enough"] and len(result["kept"]) == 6 and len(result["dropped"]) == 3
    info = json.loads((raw / "negatives" / ".done").read_text())
    assert info["screened_with"] == "budget_v1" and info["dropped_by_screen"] == 3 and info["images"] == 6
    assert len(list((raw / "negatives" / "images" / "train").iterdir())) == 6
    # Done with the same screen: skipped. Done without a screen (older run): redone.
    assert dl.download_negatives(raw, n=6, staging=tmp_path / "staging", screen=screen,
                                 screen_name="budget_v1")["kept"] == []
    sizes.clear()
    many, _ = fake_screen({f"{i:03d}.jpg": [("stairs", 0.9, 0, 0, 1, 1)] for i in range(12) if i % 2})
    result = dl.download_negatives(raw, n=8, staging=tmp_path / "staging", screen=many, screen_name="other")
    assert not result["enough"] and len(result["kept"]) == 6  # only 6 clean images exist
    assert sizes == [32, 64]  # asked for more, got no new images, stopped


def test_draw_hits_marks_detections(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), np.zeros((100, 200, 3), np.uint8))
    out = dl.draw_hits(path, [("stairs", 0.5, 0.25, 0.5, 0.75, 0.9)])
    assert out.shape == (100, 200, 3) and out[50, 100, 0] > 200  # red (RGB) top edge of the box
