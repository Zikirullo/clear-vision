"""Tests for camera hiccups: src/camera.py (riding out short gaps) and src/main.py
(CameraSupervisor reconnects with one capture at a time, "Camera lost"/"Camera back" once).

A fake cv2.VideoCapture plays a script of frames and gaps; no webcam needed.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.camera as camera_mod
import src.main as main_mod
from src.camera import Camera

FRAME = np.zeros((4, 4, 3), np.uint8)


class ScriptedCapture:
    """Fake webcam: delivers frames, except while `gap()` says the camera is silent.
    A failed read returns (False, None), as OpenCV does on a hiccup."""

    def __init__(self, source: Any) -> None:
        self.opened_at = time.monotonic()
        self.gaps: list[tuple[float, float]] = ScriptedCapture.gaps
        self.released = False

    gaps: list[tuple[float, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return True

    def set(self, prop: int, value: float) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 30.0 if prop == camera_mod.cv2.CAP_PROP_FPS else 640.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(1 / 100)  # 100 fps keeps the test quick
        t = time.monotonic() - self.opened_at
        if any(start <= t < end for start, end in self.gaps):
            return False, None
        return True, FRAME

    def release(self) -> None:
        self.released = True


def sample_health(cam: Camera, seconds: float) -> list[tuple[float, bool]]:
    start, out = time.monotonic(), []
    while (t := time.monotonic() - start) < seconds:
        out.append((t, cam.healthy))
        time.sleep(0.01)
    return out


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedCapture]:
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", ScriptedCapture)
    return ScriptedCapture


def test_short_hiccup_is_ridden_out(scripted: type[ScriptedCapture]) -> None:
    scripted.gaps = [(0.3, 0.6)]  # 0.3 s without frames, threshold 0.5 s
    cam = Camera(0, stale_timeout=0.5).start()
    try:
        health = sample_health(cam, 1.2)
    finally:
        cam.stop()
    assert all(ok for _, ok in health[5:]), "a 0.3 s hiccup must not mark the camera lost"


def test_long_gap_marks_camera_lost_then_back(scripted: type[ScriptedCapture]) -> None:
    scripted.gaps = [(0.2, 1.2)]  # 1 s without frames, threshold 0.5 s
    cam = Camera(0, stale_timeout=0.5).start()
    try:
        health = sample_health(cam, 1.8)
    finally:
        cam.stop()
    lost = [t for t, ok in health if not ok]
    assert lost, "a 1 s gap must mark the camera lost"
    assert 0.6 <= lost[0] <= 0.9  # ~0.5 s after the last frame (at 0.2 s)
    assert health[-1][1], "frames came back, so the camera is healthy again"


def test_default_threshold_is_one_and_a_half_seconds() -> None:
    assert camera_mod.DEFAULT_STALE_TIMEOUT_S == 1.5
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    assert cfg["camera"]["stale_timeout"] == 1.5


# ---------------------------------------------------------------------- supervisor: one capture at a time

class FakeCamera:
    """Counts how many cameras are open at once."""

    open_now = 0
    max_open = 0
    lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.running = False
        self.ended = False

    def start(self) -> FakeCamera:
        with FakeCamera.lock:
            FakeCamera.open_now += 1
            FakeCamera.max_open = max(FakeCamera.max_open, FakeCamera.open_now)
        self.running = True
        return self

    def stop(self) -> None:
        if self.running:
            with FakeCamera.lock:
                FakeCamera.open_now -= 1
            self.running = False

    @property
    def healthy(self) -> bool:
        return False  # always dead: forces reconnects


def test_reconnect_releases_the_old_camera_before_opening_a_new_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_mod, "Camera", FakeCamera)
    FakeCamera.open_now = FakeCamera.max_open = 0
    sup = main_mod.CameraSupervisor({}, 0, retry_interval=0.0)
    assert sup.open_initial()
    for _ in range(5):  # five reconnect rounds
        sup.poll()
        for _ in range(100):
            if sup._new_camera is not None:
                break
            time.sleep(0.005)
        sup.poll()  # swap in the new camera
    sup.stop()
    assert FakeCamera.max_open == 1, "two captures on one webcam make macOS kill the new one"


# ---------------------------------------------------------------------- announcements

class FakeSupervisor:
    def __init__(self, states: list[bool]) -> None:
        self.states = states
        self.camera = type("Cam", (), {"ended": False})()

    def poll(self) -> None:
        pass

    @property
    def healthy(self) -> bool:
        return self.states.pop(0)


def test_camera_lost_and_back_are_spoken_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_mod, "LOG_DIR", tmp_path)
    args = argparse.Namespace(source=None, config=None, no_window=True, mute=True, backend=None)
    app = main_mod.App(args, {})
    spoken: list[str] = []
    monkeypatch.setattr(app, "system_message", lambda text, priority="normal": spoken.append(text))
    app.cameras = FakeSupervisor([True, True, False, False, False, True, True, True])  # type: ignore[assignment]
    try:
        for _ in range(8):
            app._check_camera()
    finally:
        app.session.close()
    assert spoken == ["Camera lost", "Camera back"]


def test_sigterm_stops_like_ctrl_c() -> None:
    with pytest.raises(KeyboardInterrupt):
        main_mod._stop_on_sigterm(signal.SIGTERM, None)
