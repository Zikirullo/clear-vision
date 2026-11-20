"""Tests for scripts/export_coreml.py: box matching, the match share, guard messages (no export)."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.export_coreml as ec


def box(cls: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> ec.Box:
    return (cls, conf, x1, y1, x2, y2)


def test_match_boxes_same_class_iou_over_half_one_to_one() -> None:
    pt = [box("car", 0.9, 0.1, 0.1, 0.3, 0.3), box("car", 0.5, 0.6, 0.6, 0.8, 0.8), box("pole", 0.4, 0.4, 0.0, 0.45, 1.0)]
    cm = [box("car", 0.88, 0.11, 0.1, 0.31, 0.3),     # pairs with the first car
          box("truck", 0.5, 0.6, 0.6, 0.8, 0.8),     # right place, other class: no pair
          box("pole", 0.3, 0.46, 0.0, 0.51, 1.0),    # IoU 0 with the pole
          box("car", 0.87, 0.105, 0.1, 0.305, 0.3)]  # second partner for the same car: not counted twice
    assert ec.match_boxes(pt, cm) == 1


def test_match_share_counts_missing_and_extra_boxes() -> None:
    assert ec.match_share(47, 49, 48) == pytest.approx(94 / 97)  # budget_v1 on the street photos: 96.9%
    assert ec.match_share(0, 0, 0) == 1.0
    assert ec.match_share(5, 5, 10) == pytest.approx(10 / 15)


def test_missing_weights_says_where_to_get_them(capsys: pytest.CaptureFixture[str]) -> None:
    assert ec.main(["--run", "no_such_run"]) == 1
    out = capsys.readouterr().out
    assert "MyDrive/clear_vision/exports/no_such_run/best.pt" in out and "models/no_such_run/best.pt" in out


def test_refuses_numpy_2_4(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import numpy

    if not (ec.ROOT / "models" / "budget_v1" / "best.pt").exists():
        pytest.skip("models/budget_v1/best.pt is not here (gitignored)")
    monkeypatch.setattr(numpy, "__version__", "2.4.1")
    assert ec.main(["--run", "budget_v1", "--skip-export"]) == 1
    assert "issue #2633" in capsys.readouterr().out


def test_list_images(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in ec.list_images([tmp_path])] == ["a.jpg", "b.png"]
    assert len(ec.list_images([tmp_path], limit=1)) == 1
