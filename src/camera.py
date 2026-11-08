"""Threaded camera capture that always serves the latest frame.

A background thread reads from cv2.VideoCapture as fast as the source delivers
frames and keeps only the newest one, so the main loop never processes stale,
lagging frames. Works with webcam indices, video files and stream URLs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from types import TracebackType

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STALE_TIMEOUT_S = 2.0
_RETRY_DELAY_S = 0.05


def _normalize_source(source: int | str) -> int | str:
    """Turn a numeric string like "0" into a webcam index."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


def _is_file(source: int | str) -> bool:
    return isinstance(source, str) and os.path.isfile(source)


class Camera:
    """Background frame grabber that keeps only the most recent frame.

    Args:
        source: Webcam index (0, 1, ...), video file path, or stream URL.
        width: Requested capture width (webcams/streams only; may be ignored).
        height: Requested capture height (webcams/streams only; may be ignored).
        stale_timeout: Seconds without a new frame before `healthy` turns False.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        stale_timeout: float = DEFAULT_STALE_TIMEOUT_S,
    ) -> None:
        self.source = _normalize_source(source)
        self.width = width
        self.height = height
        self.stale_timeout = stale_timeout
        self.is_file = _is_file(self.source)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._timestamp: float | None = None
        self._started_at: float | None = None
        self._ended = False
        self.source_fps: float = 0.0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> Camera:
        """Open the source and start the capture thread.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        if self._thread is not None and self._thread.is_alive():
            return self

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera source: {self.source!r}")

        if not self.is_file:
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Ask the backend not to queue frames; not every backend honours it,
            # which is why the reader thread drains frames continuously anyway.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps and fps > 0 else 0.0
        self._cap = cap
        self._ended = False
        self._stop_event.clear()
        self._started_at = time.monotonic()

        logger.info(
            "Camera opened: source=%r size=%dx%d fps=%.1f file=%s",
            self.source,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.source_fps,
            self.is_file,
        )

        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the capture thread and release the source."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped: source=%r", self.source)

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------ access

    def read(self) -> tuple[np.ndarray, float] | tuple[None, None]:
        """Return the latest (frame, timestamp), or (None, None) if none yet.

        The timestamp is `time.monotonic()` at capture, for latency measurement.
        The same frame is returned until a newer one arrives; compare timestamps
        to detect new frames. Treat the frame as read-only.
        """
        with self._lock:
            if self._frame is None or self._timestamp is None:
                return None, None
            return self._frame, self._timestamp

    @property
    def healthy(self) -> bool:
        """False if no new frame has arrived for `stale_timeout` seconds.

        Computed on access, so it still turns False when the backend's read()
        call hangs (e.g. an unplugged webcam or dropped stream).
        """
        with self._lock:
            last = self._timestamp if self._timestamp is not None else self._started_at
        if last is None:
            return False
        return time.monotonic() - last < self.stale_timeout

    @property
    def ended(self) -> bool:
        """True once a video file has played to the end."""
        return self._ended

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        assert self._cap is not None
        cap = self._cap
        # Video files decode far faster than real time, so pace them to the
        # file's FPS. Live sources are paced by the device itself.
        interval = 1.0 / self.source_fps if self.is_file and self.source_fps else 0.0
        next_due = time.monotonic()
        failures = 0

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            now = time.monotonic()

            if not ok or frame is None:
                if self.is_file:
                    logger.info("Video file ended: %r", self.source)
                    self._ended = True
                    return
                failures += 1
                if failures == 1:
                    logger.warning("Camera read failed: source=%r", self.source)
                time.sleep(_RETRY_DELAY_S)
                continue

            if failures:
                logger.info("Camera recovered after %d failed reads", failures)
                failures = 0

            if interval:
                next_due += interval
                delay = next_due - now
                if delay > 0:
                    self._stop_event.wait(delay)
                    now = time.monotonic()
                elif delay < -interval:
                    # Fell behind (e.g. process paused); resync instead of racing.
                    next_due = now

            with self._lock:
                self._frame = frame
                self._timestamp = now
