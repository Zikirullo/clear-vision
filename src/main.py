"""Clear Vision: camera -> detector -> navigator -> speaker.

Run from the repo root:
    python -m src.main                       # webcam from config
    python -m src.main --source clip.mp4     # recorded walk
    python -m src.main --no-window --mute    # headless, log only

Press q in the window (or Ctrl+C in the terminal) to quit.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import yaml

from src.camera import Camera
from src.detection import Detection
from src.detector import Detector
from src.navigator import Message, NavigatorConfig, choose_message, direction, proximity, risk
from src.speaker import Speaker

logger = logging.getLogger("clear_vision")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
LOG_DIR = ROOT / "logs"

WINDOW = "Clear Vision"
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)


# ---------------------------------------------------------------------- setup

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear Vision navigation assistant")
    parser.add_argument("--source", help="webcam index, video path or stream URL (default: config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--no-window", action="store_true", help="run without the debug window")
    parser.add_argument("--mute", action="store_true", help="don't speak (messages are still logged)")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


class SessionLog:
    """CSV log of every spoken (or, when muted, would-be spoken) message."""

    FIELDS = ["time", "elapsed_s", "priority", "key", "text", "frame_to_decision_ms", "muted"]

    def __init__(self, directory: Path, muted: bool) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.muted = muted
        self._start = time.monotonic()
        self._file: TextIO = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, msg: Message, frame_to_decision_ms: float | None = None) -> None:
        self._writer.writerow({
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": f"{time.monotonic() - self._start:.3f}",
            "priority": msg.priority,
            "key": msg.key,
            "text": msg.text,
            "frame_to_decision_ms": "" if frame_to_decision_ms is None else f"{frame_to_decision_ms:.1f}",
            "muted": self.muted,
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------- camera supervision

class CameraSupervisor:
    """Owns the Camera and reopens it in the background when it goes dead.

    Opening a webcam or stream can block for seconds, so reconnects (and
    stopping a hung camera) happen on helper threads, never in the main loop.
    """

    def __init__(self, cam_cfg: dict[str, Any], source: int | str, retry_interval: float) -> None:
        self.cam_cfg = cam_cfg
        self.source = source
        self.retry_interval = retry_interval
        self.camera: Camera | None = None
        self._lock = threading.Lock()
        self._new_camera: Camera | None = None
        self._reconnecting = False
        self._last_attempt = -float("inf")

    def _make_camera(self) -> Camera:
        return Camera(
            self.source,
            self.cam_cfg.get("width"),
            self.cam_cfg.get("height"),
            stale_timeout=self.cam_cfg.get("stale_timeout", 2.0),
        )

    def open_initial(self) -> bool:
        try:
            self.camera = self._make_camera().start()
            return True
        except RuntimeError as exc:
            logger.error("%s", exc)
            self._last_attempt = time.monotonic()
            return False

    @property
    def healthy(self) -> bool:
        return self.camera is not None and self.camera.healthy

    def poll(self) -> None:
        """Swap in a reconnected camera, or start a reconnect if one is due."""
        with self._lock:
            new, self._new_camera = self._new_camera, None
        if new is not None:
            old, self.camera = self.camera, new
            if old is not None:
                threading.Thread(target=old.stop, daemon=True).start()
            logger.info("Camera reconnected")
            return

        if self.healthy or (self.camera is not None and self.camera.ended):
            return
        now = time.monotonic()
        with self._lock:
            if self._reconnecting or now - self._last_attempt < self.retry_interval:
                return
            self._reconnecting = True
            self._last_attempt = now
        threading.Thread(target=self._reconnect, name="camera-reconnect", daemon=True).start()

    def _reconnect(self) -> None:
        logger.info("Trying to reopen camera %r", self.source)
        try:
            camera = self._make_camera().start()
        except RuntimeError as exc:
            logger.warning("Reconnect failed: %s", exc)
            camera = None
        with self._lock:
            self._new_camera = camera
            self._reconnecting = False

    def stop(self) -> None:
        if self.camera is not None:
            self.camera.stop()
        with self._lock:
            if self._new_camera is not None:
                self._new_camera.stop()


# ---------------------------------------------------------------------- drawing

@dataclass
class Timings:
    fps: float = 0.0
    capture_age_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    history: list[float] = field(default_factory=list)


def box_color(det: Detection, nav_cfg: NavigatorConfig) -> tuple[int, int, int]:
    if direction(det, nav_cfg) == "ahead" and proximity(det, nav_cfg) == "near":
        return RED
    if risk(det, nav_cfg) >= nav_cfg.risk_threshold:
        return ORANGE
    return GREEN


def draw_detections(frame: np.ndarray, detections: list[Detection], nav_cfg: NavigatorConfig) -> None:
    h, w = frame.shape[:2]
    for x in (nav_cfg.left_boundary, nav_cfg.right_boundary):
        cv2.line(frame, (int(x * w), 0), (int(x * w), h), GRAY, 1)
    for det in detections:
        color = box_color(det, nav_cfg)
        p1 = (int(det.x1 * w), int(det.y1 * h))
        p2 = (int(det.x2 * w), int(det.y2 * h))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{det.class_name} {det.confidence:.2f} {direction(det, nav_cfg)} {proximity(det, nav_cfg)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(p1[1] - th - 6, 0)
        cv2.rectangle(frame, (p1[0], top), (p1[0] + tw + 4, top + th + 6), color, -1)
        cv2.putText(frame, label, (p1[0] + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray, t: Timings, device: str, healthy: bool, last_msg: Message | None
) -> None:
    lines = [
        f"FPS {t.fps:4.1f}   device {device}   camera {'OK' if healthy else 'LOST'}",
        f"capture age {t.capture_age_ms:5.1f} ms   inference {t.inference_ms:5.1f} ms   total {t.total_ms:5.1f} ms",
    ]
    for i, text in enumerate(lines):
        y = 24 + i * 24
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    if last_msg is not None:
        color = RED if last_msg.priority == "urgent" else ORANGE
        h = frame.shape[0]
        cv2.putText(frame, last_msg.text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, last_msg.text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------- app

class App:
    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
        self.args = args
        self.cfg = cfg
        app_cfg = cfg.get("app", {})
        self.show_message_s: float = app_cfg.get("show_message_s", 2.0)
        self.stats_interval_s: float = app_cfg.get("stats_interval_s", 5.0)

        cam_cfg = cfg.get("camera", {})
        source = args.source if args.source is not None else cam_cfg.get("source", 0)
        self.cameras = CameraSupervisor(cam_cfg, source, app_cfg.get("camera_retry_s", 2.0))
        self.nav_cfg = NavigatorConfig.from_dict(cfg.get("navigator", {}))
        self.speaker: Speaker | None = None if args.mute else Speaker.from_config(cfg.get("speaker", {}))
        self.session = SessionLog(LOG_DIR, muted=args.mute)
        self.detector: Detector | None = None

        self.last_spoken: dict[str, float] = {}
        self.camera_lost_announced = False
        self.last_msg: Message | None = None
        self.last_msg_time = 0.0
        self.timings = Timings()

    # ------------------------------------------------------------------ speech

    def announce(self, msg: Message, frame_to_decision_ms: float | None = None) -> None:
        if self.speaker is not None:
            self.speaker.say(msg)
        self.session.write(msg, frame_to_decision_ms)
        self.last_msg, self.last_msg_time = msg, time.monotonic()
        logger.info("Say [%s]: %s", msg.priority, msg.text)

    def system_message(self, text: str, priority: str = "normal") -> None:
        self.announce(Message(text=text, priority=priority, key=f"system:{text}"))  # type: ignore[arg-type]

    @property
    def speaker_busy(self) -> bool:
        return self.speaker is not None and self.speaker.is_busy

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        if self.speaker is not None:
            self.speaker.start()
        try:
            self.detector = Detector.from_config(self.cfg.get("detector", {}))
        except Exception:
            logger.exception("Could not load the detection model")
            self.system_message("Detection model failed. Clear Vision cannot run.", "urgent")
            self._wait_for_speech(5.0)
            return 1

        self.system_message("Clear Vision started")
        if not self.cameras.open_initial():
            self.system_message("Camera not available", "urgent")
            self.camera_lost_announced = True
        logger.info("Session log: %s", self.session.path)

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Interrupted (Ctrl+C)")
        return 0

    def _loop(self) -> None:
        assert self.detector is not None
        last_ts: float | None = None
        last_stats = time.monotonic()
        while True:
            self._check_camera()
            camera = self.cameras.camera
            if camera is not None and camera.ended:
                logger.info("Video ended")
                return

            frame, ts = camera.read() if camera is not None else (None, None)
            if frame is not None and ts is not None and ts != last_ts:
                self._process(frame, ts, last_ts)
                last_ts = ts
            elif self.args.no_window:
                time.sleep(0.002)

            if not self.args.no_window and cv2.waitKey(1) & 0xFF == ord("q"):
                return
            now = time.monotonic()
            if now - last_stats >= self.stats_interval_s:
                self._log_stats()
                last_stats = now

    def _check_camera(self) -> None:
        self.cameras.poll()
        healthy = self.cameras.healthy
        if not healthy and not self.camera_lost_announced:
            camera = self.cameras.camera
            if camera is None or not camera.ended:
                self.system_message("Camera lost", "urgent")
                self.camera_lost_announced = True
        elif healthy and self.camera_lost_announced:
            self.system_message("Camera back")
            self.camera_lost_announced = False

    def _process(self, frame: np.ndarray, ts: float, last_ts: float | None) -> None:
        assert self.detector is not None
        t = self.timings
        t.capture_age_ms = (time.monotonic() - ts) * 1000
        detections = self.detector.detect(frame)
        t.inference_ms = self.detector.last_inference_ms

        now = time.monotonic()
        msg = choose_message(detections, now, self.last_spoken, self.nav_cfg)
        t.total_ms = (time.monotonic() - ts) * 1000
        t.history.append(t.total_ms)
        # Normal messages wait for silence instead of piling up; urgent ones
        # always go through (the speaker interrupts for them).
        if msg is not None and (msg.priority == "urgent" or not self.speaker_busy):
            self.last_spoken[msg.key] = now
            self.announce(msg, t.total_ms)

        if last_ts is not None and ts > last_ts:
            # Exponential moving average of processed frames per second.
            fps = 1.0 / (ts - last_ts)
            t.fps = fps if t.fps == 0 else 0.9 * t.fps + 0.1 * fps

        if not self.args.no_window:
            shown = frame.copy()
            draw_detections(shown, detections, self.nav_cfg)
            recent = self.last_msg if now - self.last_msg_time < self.show_message_s else None
            draw_overlay(shown, t, self.detector.device, self.cameras.healthy, recent)
            cv2.imshow(WINDOW, shown)

    def _log_stats(self) -> None:
        t = self.timings
        if t.history:
            avg = sum(t.history) / len(t.history)
            logger.info(
                "FPS %.1f | capture age %.1f ms | inference %.1f ms | frame->decision avg %.1f ms max %.1f ms",
                t.fps, t.capture_age_ms, t.inference_ms, avg, max(t.history),
            )
            t.history.clear()

    def _wait_for_speech(self, timeout: float) -> None:
        if self.speaker is not None:
            self.speaker.wait_until_idle(timeout)

    def shutdown(self) -> None:
        logger.info("Shutting down")
        self.cameras.stop()
        if self.speaker is not None:
            self.speaker.stop()
        self.session.close()
        if not self.args.no_window:
            cv2.destroyAllWindows()
        logger.info("Session log saved: %s", self.session.path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    app = App(args, cfg)
    try:
        return app.run()
    except KeyboardInterrupt:
        return 0
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())
