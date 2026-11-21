"""Turn a folder of photos into a slideshow video (each photo shown for a few seconds).

    python scripts/make_photo_video.py data/street_photos street.mp4        # 24 photos -> 48 s
    python scripts/make_photo_video.py data/street_photos street.mp4 --seconds 3

Used for repeatable checks: play it full screen on the Mac and point the phone at it, or
feed it to scripts/replay.py. Photos are letterboxed (black bars) into the frame size,
in file-name order.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
log = logging.getLogger("make_photo_video")


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit `image` inside width x height without distortion, padding with black."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def make_video(photos: Sequence[Path], out: Path, seconds: float = 2.0, fps: int = 30,
               width: int = 1280, height: int = 720) -> int:
    """Write the slideshow; returns the number of photos used (unreadable files skipped)."""
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {out}")
    used = 0
    try:
        for path in photos:
            image = cv2.imread(str(path))
            if image is None:
                log.warning("skipped unreadable %s", path.name)
                continue
            frame = letterbox(image, width, height)
            for _ in range(round(seconds * fps)):
                writer.write(frame)
            used += 1
    finally:
        writer.release()
    return used


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=2.0, help="seconds per photo (default 2)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)
    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        log.error("no photos in %s", args.folder)
        return 2
    used = make_video(photos, args.out, args.seconds, args.fps)
    log.info("wrote %s: %d photos, %.0f s", args.out, used, used * args.seconds)
    return 0 if used else 2


if __name__ == "__main__":
    sys.exit(main())
