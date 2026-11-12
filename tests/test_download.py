"""Tests for scripts/download_datasets.py (shared by notebooks 01 and 02), with a fake Roboflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    assert 'DOWNLOAD_NEW = ["rf_sidewalk", "rf_barricade"]' in code[0]
    download = next(i for i, c in enumerate(code) if "download_roboflow_datasets(" in c)
    merge = next(i for i, c in enumerate(code) if "merge_datasets.append(" in c)
    assert download < merge
    assert "Path(LOCAL_RAW)" in code[download]
