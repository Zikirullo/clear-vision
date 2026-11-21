"""scripts/make_photo_video.py: slideshow length, letterboxing, unreadable files."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_photo_video as mpv  # noqa: E402


def test_letterbox_keeps_aspect_and_pads_black() -> None:
    tall = np.full((400, 100, 3), 255, np.uint8)
    out = mpv.letterbox(tall, 320, 180)
    assert out.shape == (180, 320, 3)
    assert out[:, :100].max() == 0 and out[90, 160].min() == 255  # bars left, photo centered


def test_video_has_seconds_per_photo(tmp_path: Path) -> None:
    for i, color in enumerate((0, 128)):
        cv2.imwrite(str(tmp_path / f"p{i}.png"), np.full((90, 160, 3), color, np.uint8))
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "v.mp4"
    photos = sorted(tmp_path.glob("*.png")) + [tmp_path / "broken.jpg"]
    assert mpv.make_video(photos, out, seconds=1.0, fps=10, width=160, height=90) == 2
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames == 20


def test_main_reports_empty_folder(tmp_path: Path) -> None:
    assert mpv.main([str(tmp_path), str(tmp_path / "v.mp4")]) == 2
