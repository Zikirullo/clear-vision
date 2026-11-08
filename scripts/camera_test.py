"""Show the live camera feed with FPS overlaid. Press q to quit.

Usage:
    python scripts/camera_test.py                 # default webcam
    python scripts/camera_test.py 1               # another webcam index
    python scripts/camera_test.py path/to/video.mp4
    python scripts/camera_test.py http://192.168.1.20:8080/video
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera  # noqa: E402

WINDOW = "Clear Vision - camera test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default="0", help="webcam index, video path or URL")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    try:
        camera = Camera(args.source, args.width, args.height).start()
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    fps = 0.0
    last_ts: float | None = None
    try:
        while True:
            frame, ts = camera.read()

            if frame is not None and ts != last_ts:
                if last_ts is not None:
                    dt = ts - last_ts
                    if dt > 0:
                        # Exponential moving average keeps the number readable.
                        fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
                last_ts = ts

                shown = frame.copy()
                status = "OK" if camera.healthy else "NO SIGNAL"
                cv2.putText(shown, f"{fps:5.1f} FPS  {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, shown)
            elif not camera.healthy and frame is not None:
                shown = frame.copy()
                cv2.putText(shown, "NO SIGNAL", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, shown)

            if camera.ended:
                logging.info("Playback finished")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if frame is None:
                time.sleep(0.005)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
