"""Tests for the speech filter (src/navigator.py plan_speech, src/planner.py) and scripts/replay.py.

Fake detections only: no camera, model or audio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.detection import Detection
from src.navigator import (
    NavigatorConfig, in_speaking_zone, is_confirmed, name_changed_recently, name_is_doubtful, same_object,
    speak_threshold,
)
from src.planner import SpeechPlanner

ROOT = Path(__file__).resolve().parent.parent
FPS = 10.0


@pytest.fixture(scope="module")
def cfg() -> NavigatorConfig:
    return NavigatorConfig.from_dict(yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["navigator"])


def box(name: str = "stairs", conf: float = 0.8, cx: float = 0.5, top: float = 0.4, h: float = 0.5,
        w: float = 0.3) -> Detection:
    """Box centered at `cx`, from `top` down `h` (default: big, ahead, close)."""
    return Detection(name, conf, cx - w / 2, top, cx + w / 2, top + h)


def run(planner: SpeechPlanner, frames: list[list[Detection]], start: float = 0.0) -> list[tuple[float, str]]:
    """Feed frames at FPS; every planned message counts as spoken. Returns (time, text)."""
    spoken = []
    for i, dets in enumerate(frames):
        now = start + i / FPS
        plan = planner.update(dets, now)
        if plan.message is not None:
            planner.mark_spoken(plan.message, now)
            spoken.append((round(now, 2), plan.message.text))
    return spoken


# ---------------------------------------------------------------------- config

def test_config_values(cfg: NavigatorConfig) -> None:
    assert (cfg.confirm_frames, cfg.window_frames, cfg.confirm_center_dist, cfg.confirm_iou) == (3, 5, 0.15, 0.3)
    assert (cfg.cooldown, cfg.urgent_cooldown) == (4.0, 2.0)
    assert [speak_threshold(c, cfg) for c in ("car", "bus", "truck", "motorcycle")] == [0.45] * 4
    assert speak_threshold("person", cfg) == 0.35
    assert [speak_threshold(c, cfg) for c in ("pole", "bollard", "traffic_cone", "fire_hydrant")] == [0.25] * 4
    assert [speak_threshold(c, cfg) for c in ("curb", "stairs", "pothole", "manhole")] == [0.20] * 4
    assert speak_threshold("bench", cfg) == speak_threshold("dog", cfg) == 0.35
    assert cfg.mute_non_hazards is False


# ---------------------------------------------------------------------- confirmation

def test_flicker_one_of_five_frames_stays_silent(cfg: NavigatorConfig) -> None:
    frames = ([[box("car", 0.9, h=0.6)]] + [[]] * 4) * 6  # a "car" in 1 of every 5 frames, for 3 s
    assert run(SpeechPlanner(cfg), frames) == []


def test_two_of_five_stays_silent_three_of_five_speaks(cfg: NavigatorConfig) -> None:
    s = box("pole", 0.6, cx=0.8, w=0.08)
    assert run(SpeechPlanner(cfg), [[s], [], [s], [], [], []]) == []
    spoken = run(SpeechPlanner(cfg), [[s], [], [s], [], [s]])
    assert spoken == [(0.4, "Pole, right")]  # third sighting within 5 frames, gaps allowed


def test_steady_object_is_spoken_once(cfg: NavigatorConfig) -> None:
    frames = [[box("stairs", 0.7, cx=0.2, w=0.2, h=0.3, top=0.6)]] * 35  # 3.5 s, left, mid distance
    assert run(SpeechPlanner(cfg), frames) == [(0.2, "Stairs, left")]  # confirmed on frame 3, then cooldown


def test_slowly_moving_object_stays_confirmed(cfg: NavigatorConfig) -> None:
    frames = [[box("person", 0.8, cx=0.40 + 0.02 * i, w=0.15)] for i in range(5)]
    assert run(SpeechPlanner(cfg), frames)[0][0] == 0.2


def test_jumping_boxes_are_not_one_object(cfg: NavigatorConfig) -> None:
    # Same class, but each frame somewhere else (as when walls become "poles" here and there).
    frames = [[box("pole", 0.9, cx=x, w=0.06)] for x in (0.1, 0.5, 0.9, 0.3, 0.7, 0.1, 0.5)]
    assert run(SpeechPlanner(cfg), frames) == []


def test_other_class_in_the_same_place_does_not_confirm(cfg: NavigatorConfig) -> None:
    a, b = box("car", 0.9), box("truck", 0.9)
    assert not same_object(a, b, cfg)
    assert not is_confirmed(a, [[b], [b], [b], [b]], cfg)
    assert is_confirmed(a, [[b], [a], [b], [a]], cfg)


def test_post_group_classes_confirm_each_other(cfg: NavigatorConfig) -> None:
    # The model flips between cone and hydrant on one object: still one object, so the
    # warning is never lost to a name doubt.
    cone, hydrant = box("traffic_cone", 0.9), box("fire_hydrant", 0.9)
    assert same_object(cone, hydrant, cfg)
    assert is_confirmed(cone, [[hydrant], [cone]], cfg)


def test_name_changed_recently_within_the_window(cfg: NavigatorConfig) -> None:
    cone, hydrant = box("traffic_cone", 0.9), box("fire_hydrant", 0.9)
    assert name_changed_recently(cone, [[hydrant], [cone], [cone], [cone]], cfg)  # 4 frames back: inside
    assert not name_changed_recently(cone, [[hydrant], [cone], [cone], [cone], [cone]], cfg)  # 5 back: outside
    assert name_is_doubtful(cone, [cone], cfg, history=[[hydrant], [cone]])
    assert not name_is_doubtful(cone, [cone], cfg, history=[[cone], [cone]])


def test_name_change_ignores_other_objects_and_other_groups(cfg: NavigatorConfig) -> None:
    cone = box("traffic_cone", 0.9, cx=0.2, w=0.1)
    far_hydrant = box("fire_hydrant", 0.9, cx=0.8, w=0.1)  # a different post elsewhere
    assert not name_changed_recently(cone, [[far_hydrant], [far_hydrant]], cfg)
    car, truck = box("car", 0.9), box("truck", 0.9)  # not a name group
    assert not name_is_doubtful(car, [car], cfg, history=[[truck], [truck]])


def test_name_history_one_frame_turns_the_check_off(cfg: NavigatorConfig) -> None:
    import dataclasses
    off = dataclasses.replace(cfg, name_history_frames=1)
    cone, hydrant = box("traffic_cone", 0.9), box("fire_hydrant", 0.9)
    assert not name_changed_recently(cone, [[hydrant], [hydrant]], off)


def test_planner_keeps_enough_history_for_the_name_window(cfg: NavigatorConfig) -> None:
    import dataclasses
    assert SpeechPlanner(cfg).history.maxlen == 4
    assert SpeechPlanner(dataclasses.replace(cfg, name_history_frames=8)).history.maxlen == 7
    # Confirmation still uses only the last window_frames - 1 frames.
    longer = dataclasses.replace(cfg, name_history_frames=8)
    a = box()
    assert not is_confirmed(a, [[a], [a], [], [], [], []], longer)


def test_only_the_last_window_counts(cfg: NavigatorConfig) -> None:
    a = box()
    old = [[a], [a], [], [], [], []]  # two sightings, but more than 4 frames ago
    assert not is_confirmed(a, old, cfg)


# ---------------------------------------------------------------------- cooldown

def test_normal_repeat_after_four_seconds(cfg: NavigatorConfig) -> None:
    far_left = [box("stairs", 0.7, cx=0.2, w=0.2, h=0.3, top=0.6)]  # normal priority
    spoken = run(SpeechPlanner(cfg), [far_left] * 50)  # 5 s
    assert [t for t, _ in spoken] == [0.2, 4.2]


def test_urgent_repeat_after_two_seconds(cfg: NavigatorConfig) -> None:
    close_ahead = [box("stairs", 0.7, cx=0.5, h=0.6, top=0.4)]  # urgent: ahead + near
    spoken = run(SpeechPlanner(cfg), [close_ahead] * 50)
    assert [t for t, _ in spoken] == [0.2, 2.2, 4.2]
    assert spoken[0][1] == "Stairs ahead, close"


# ---------------------------------------------------------------------- thresholds and zone

@pytest.mark.parametrize(("name", "conf", "spoken"), [
    ("car", 0.44, False), ("car", 0.46, True),
    ("person", 0.34, False), ("person", 0.36, True),
    ("pole", 0.24, False), ("pole", 0.26, True),
    ("curb", 0.19, False), ("curb", 0.21, True),
    ("bench", 0.34, False), ("bench", 0.36, True),
])
def test_speak_thresholds(cfg: NavigatorConfig, name: str, conf: float, spoken: bool) -> None:
    frames = [[box(name, conf, h=0.6, top=0.35)]] * 5
    assert bool(run(SpeechPlanner(cfg), frames)) is spoken


def test_weak_detections_are_kept_for_drawing_but_not_candidates(cfg: NavigatorConfig) -> None:
    weak = box("curb", 0.16)  # detected (fine-tuned threshold 0.15) but under the speech threshold 0.20
    plan = SpeechPlanner(cfg).update([weak], 0.0)
    assert plan.candidates == () and plan.speakable == ()


def test_far_small_objects_near_the_horizon_stay_silent(cfg: NavigatorConfig) -> None:
    near_horizon = box("pole", 0.9, top=0.2, h=0.1, w=0.03)  # bottom at 0.3, 10% tall
    assert not in_speaking_zone(near_horizon, cfg)
    assert run(SpeechPlanner(cfg), [[near_horizon]] * 10) == []
    assert in_speaking_zone(box("pole", 0.9, top=0.3, h=0.12), cfg)  # bottom at 0.42: lower 60%
    assert in_speaking_zone(box("pole", 0.9, top=0.05, h=0.2), cfg)  # bottom high, but tall


def test_vehicles_need_twenty_percent_height(cfg: NavigatorConfig) -> None:
    assert not in_speaking_zone(box("car", 0.9, top=0.7, h=0.18), cfg)  # low in frame but small: far car
    assert in_speaking_zone(box("car", 0.9, top=0.5, h=0.25), cfg)
    assert run(SpeechPlanner(cfg), [[box("car", 0.9, top=0.7, h=0.18)]] * 10) == []


def test_hazards_only_toggle(cfg: NavigatorConfig) -> None:
    planner = SpeechPlanner(cfg)
    assert planner.toggle_mute_non_hazards() is True
    dog, stairs = box("dog", 0.9, cx=0.2, w=0.2), box("stairs", 0.9, cx=0.8, w=0.2)  # dog: not a hazard class
    assert run(planner, [[dog]] * 5) == []
    assert [text for _, text in run(planner, [[dog, stairs]] * 5, start=1.0)] == ["Stairs, right"]
    planner.toggle_mute_non_hazards()
    assert [text for _, text in run(planner, [[dog]] * 5, start=2.0)] == ["Dog, left"]


# ---------------------------------------------------------------------- replay script

class FakeDetector:
    device = "cpu"
    last_inference_ms = 1.0

    def __init__(self) -> None:
        self.frame = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.frame += 1
        if self.frame % 5 == 0:  # a flickering "car"
            return [box("car", 0.9, cx=0.2, h=0.6)]
        return [box("stairs", 0.8, cx=0.5, h=0.6, top=0.4)]  # steady stairs ahead, close


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(50):  # 5 s at 10 fps
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def test_replay_prints_messages_with_video_time(clip: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay

    monkeypatch.setattr(replay, "build_detector", lambda cfg, backend=None: FakeDetector())
    assert replay.main([str(clip)]) == 0
    out = capsys.readouterr().out
    assert "    0.20s  [urgent] Stairs ahead, close" in out
    assert "    2.20s  [urgent] Stairs ahead, close" in out
    assert "Car" not in out.split("messages in")[0]  # the flicker stays silent
    assert "3 messages in 5.0 s (50 frames processed)" in out

    assert replay.main([str(clip), "--no-filter"]) == 0
    old = capsys.readouterr().out
    assert "Car, left, close" in old and "    0.00s  [urgent] Stairs ahead, close" in old


def test_replay_step_and_source_flag(clip: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    import replay

    monkeypatch.setattr(replay, "build_detector", lambda cfg, backend=None: FakeDetector())
    assert replay.main(["--source", str(clip), "--step", "2", "--seconds", "2"]) == 0
    assert "frames processed" in capsys.readouterr().out
    args = replay.parse_args(["--source", "0"])
    assert args.source == "0" and args.source.isdigit()
