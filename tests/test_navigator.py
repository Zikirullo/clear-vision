"""Tests for src/navigator.py. No camera, model or audio needed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.detection import Detection
from src.navigator import (
    Message,
    NavigatorConfig,
    choose_message,
    direction,
    proximity,
    risk,
    severity,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> NavigatorConfig:
    with (ROOT / "configs" / "default.yaml").open() as f:
        return NavigatorConfig.from_dict(yaml.safe_load(f)["navigator"])


def det(class_name: str = "person", cx: float = 0.5, h: float = 0.3, w: float = 0.1) -> Detection:
    """Box centered horizontally at `cx` with height `h`, standing on the frame bottom."""
    return Detection(class_name, 0.9, x1=cx - w / 2, y1=1.0 - h, x2=cx + w / 2, y2=1.0)


# --------------------------------------------------------------------- direction

@pytest.mark.parametrize(
    ("cx", "expected"),
    [(0.1, "left"), (0.32, "left"), (0.5, "ahead"), (0.34, "ahead"), (0.66, "ahead"),
     (0.68, "right"), (0.9, "right")],
)
def test_direction_by_frame_thirds(cfg: NavigatorConfig, cx: float, expected: str) -> None:
    assert direction(det(cx=cx), cfg) == expected


# --------------------------------------------------------------------- proximity

@pytest.mark.parametrize(
    ("h", "expected"),
    [(0.8, "near"), (0.51, "near"), (0.5, "mid"), (0.3, "mid"), (0.25, "mid"),
     (0.24, "far"), (0.05, "far")],
)
def test_proximity_by_box_height(cfg: NavigatorConfig, h: float, expected: str) -> None:
    assert proximity(det(h=h), cfg) == expected


# --------------------------------------------------------------------- severity / risk

def test_severity_from_config_with_default(cfg: NavigatorConfig) -> None:
    assert severity("car", cfg) == 3
    assert severity("person", cfg) == 2
    assert severity("chair", cfg) == 1
    assert severity("unknown thing", cfg) == cfg.default_severity


def test_risk_higher_when_ahead(cfg: NavigatorConfig) -> None:
    assert risk(det(cx=0.5), cfg) > risk(det(cx=0.1), cfg)


def test_risk_higher_when_closer(cfg: NavigatorConfig) -> None:
    assert risk(det(h=0.7), cfg) > risk(det(h=0.3), cfg) > risk(det(h=0.1), cfg)


def test_risk_higher_for_more_severe_class(cfg: NavigatorConfig) -> None:
    assert risk(det("car"), cfg) > risk(det("person"), cfg) > risk(det("chair"), cfg)


# --------------------------------------------------------------------- choose_message

def test_no_detections_no_message(cfg: NavigatorConfig) -> None:
    assert choose_message([], now=0.0, last_spoken={}, cfg=cfg) is None


def test_low_risk_objects_are_not_announced(cfg: NavigatorConfig) -> None:
    far_chair_on_side = det("chair", cx=0.1, h=0.1)
    assert risk(far_chair_on_side, cfg) < cfg.risk_threshold
    assert choose_message([far_chair_on_side], now=0.0, last_spoken={}, cfg=cfg) is None


def test_ahead_and_near_is_urgent(cfg: NavigatorConfig) -> None:
    msg = choose_message([det("car", cx=0.5, h=0.7)], now=0.0, last_spoken={}, cfg=cfg)
    assert msg == Message(text="Car ahead, close", priority="urgent", key="car:ahead")


def test_side_object_is_normal_with_short_phrase(cfg: NavigatorConfig) -> None:
    msg = choose_message([det("person", cx=0.1, h=0.3)], now=0.0, last_spoken={}, cfg=cfg)
    assert msg == Message(text="Person, left", priority="normal", key="person:left")


def test_near_side_object_says_close_but_is_not_urgent(cfg: NavigatorConfig) -> None:
    msg = choose_message([det("bicycle", cx=0.9, h=0.6)], now=0.0, last_spoken={}, cfg=cfg)
    assert msg is not None
    assert msg.text == "Bicycle, right, close"
    assert msg.priority == "normal"


def test_picks_most_dangerous_object(cfg: NavigatorConfig) -> None:
    detections = [
        det("chair", cx=0.5, h=0.3),    # mid ahead, low severity
        det("person", cx=0.1, h=0.4),   # mid left
        det("car", cx=0.5, h=0.6),      # near ahead, high severity -> winner
        det("dog", cx=0.9, h=0.7),      # near right
    ]
    msg = choose_message(detections, now=0.0, last_spoken={}, cfg=cfg)
    assert msg is not None and msg.key == "car:ahead"


def test_cooldown_suppresses_repeat(cfg: NavigatorConfig) -> None:
    person_left = [det("person", cx=0.1, h=0.3)]
    spoken = {"person:left": 10.0}
    assert choose_message(person_left, now=10.0 + cfg.cooldown - 0.1, last_spoken=spoken, cfg=cfg) is None
    msg = choose_message(person_left, now=10.0 + cfg.cooldown + 0.1, last_spoken=spoken, cfg=cfg)
    assert msg is not None and msg.key == "person:left"


def test_cooldown_is_per_class_and_direction(cfg: NavigatorConfig) -> None:
    spoken = {"person:left": 10.0}
    msg = choose_message([det("person", cx=0.9, h=0.3)], now=10.5, last_spoken=spoken, cfg=cfg)
    assert msg is not None and msg.key == "person:right"


def test_suppressed_top_object_falls_back_to_next(cfg: NavigatorConfig) -> None:
    detections = [det("car", cx=0.1, h=0.4), det("person", cx=0.9, h=0.4)]
    msg = choose_message(detections, now=11.0, last_spoken={"car:left": 10.0}, cfg=cfg)
    assert msg is not None and msg.key == "person:right"


def test_urgent_uses_shorter_cooldown(cfg: NavigatorConfig) -> None:
    assert cfg.urgent_cooldown < cfg.cooldown
    car_close_ahead = [det("car", cx=0.5, h=0.7)]
    spoken = {"car:ahead": 10.0}
    now = 10.0 + cfg.urgent_cooldown + 0.1  # past urgent cooldown, within normal cooldown
    msg = choose_message(car_close_ahead, now=now, last_spoken=spoken, cfg=cfg)
    assert msg is not None and msg.priority == "urgent"


def test_choose_message_does_not_modify_history(cfg: NavigatorConfig) -> None:
    spoken: dict[str, float] = {}
    choose_message([det("car", cx=0.5, h=0.7)], now=5.0, last_spoken=spoken, cfg=cfg)
    assert spoken == {}


def test_navigator_imports_no_opencv_or_model() -> None:
    code = (
        "import sys; import src.navigator; "
        "bad = [m for m in ('cv2', 'torch', 'ultralytics') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
