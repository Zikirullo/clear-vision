"""Play a video (or the webcam) through the full pipeline and print every spoken message.

    python scripts/replay.py walk.mp4                    # with the speech filter (config)
    python scripts/replay.py walk.mp4 --no-filter        # the old behaviour, for before/after
    python scripts/replay.py --source 0 --seconds 60     # webcam for 60 s
    python scripts/replay.py walk.mp4 --show             # with the debug window

Detector -> speech planner -> messages, exactly as in src/main.py with --mute (nothing
is spoken aloud; every message the app would say is printed with its time).

- Video files are read frame by frame on the video's own clock (frame / fps), so cooldowns
  behave as live, and a clip always gives the same result.
- `--step N` processes every Nth frame (the live app on the M1 handles ~21 of 30 fps).
- The webcam uses the real clock.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import BACKENDS, build_detector  # noqa: E402
from src.navigator import NavigatorConfig  # noqa: E402
from src.planner import SpeechPlanner  # noqa: E402

logger = logging.getLogger("replay")


def unfiltered(cfg: NavigatorConfig) -> NavigatorConfig:
    """The behaviour before the speech filter: every detection, first frame, 1.5 s urgent repeat."""
    return dataclasses.replace(
        cfg, confirm_frames=1, window_frames=1, speak_conf={}, speak_conf_default=0.0,
        horizon_bottom=0.0, min_speak_height=0.0, vehicle_min_height=0.0, mute_non_hazards=False,
        urgent_cooldown=1.5,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print every message the app would speak for a video or the webcam")
    parser.add_argument("video", nargs="?", help="video file (or use --source)")
    parser.add_argument("--source", help="video file, or a webcam index such as 0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--backend", choices=BACKENDS, help="detector backend (default: config)")
    parser.add_argument("--no-filter", action="store_true", help="disable the speech filter (old behaviour)")
    parser.add_argument("--hazards-only", action="store_true", help="speak only hazard classes (key m live)")
    parser.add_argument("--step", type=int, default=1, help="process every Nth frame of a video file")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    parser.add_argument("--show", action="store_true", help="show the debug window (q quits)")
    args = parser.parse_args(argv)
    args.source = args.source if args.source is not None else args.video
    if args.source is None:
        parser.error("give a video file or --source")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    nav_cfg = NavigatorConfig.from_dict(cfg.get("navigator", {}))
    if args.no_filter:
        nav_cfg = unfiltered(nav_cfg)
    detector = build_detector(cfg.get("detector", {}), args.backend)
    planner = SpeechPlanner(nav_cfg)
    planner.mute_non_hazards = args.hazards_only

    webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if webcam else args.source)
    if not cap.isOpened():
        logger.error("Cannot open %s", args.source)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mode = "no filter (old behaviour)" if args.no_filter else "speech filter"
    print(f"{args.source}: {mode}, {type(detector).__name__} on {detector.device}"
          + ("" if webcam else f", {fps:.0f} fps, every {args.step} frame(s)"))

    if args.show:
        from src.main import Timings, draw_detections, draw_overlay

    start = time.monotonic()
    spoken: list[tuple[float, str, str]] = []
    frame_no = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic() - start if webcam else frame_no / fps
        frame_no += 1
        if args.seconds is not None and now > args.seconds:
            break
        if not webcam and (frame_no - 1) % args.step:
            continue
        processed += 1
        detections = detector.detect(frame)
        plan = planner.update(detections, now)
        if plan.message is not None:
            planner.mark_spoken(plan.message, now)
            spoken.append((now, plan.message.priority, plan.message.text))
            print(f"{now:8.2f}s  [{plan.message.priority:6s}] {plan.message.text}", flush=True)
        if args.show:
            draw_detections(frame, detections, nav_cfg, plan.speakable)
            draw_overlay(frame, Timings(inference_ms=detector.last_inference_ms), detector.device, True,
                         None, planner.mute_non_hazards)
            cv2.imshow("replay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    duration = time.monotonic() - start if webcam else frame_no / fps
    by_class = Counter(text.split(" ahead")[0].split(",")[0] for _, _, text in spoken)
    print(f"\n{len(spoken)} messages in {duration:.1f} s ({processed} frames processed)")
    for name, n in by_class.most_common():
        print(f"  {n:3d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
