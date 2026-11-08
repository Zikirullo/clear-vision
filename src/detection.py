"""The Detection type shared by detector, tracker and navigator.

Kept in its own dependency-free module so pure logic (navigator) and its tests
can use it without importing Ultralytics, torch or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object. Box coordinates are normalized to 0-1 of the frame."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def center_x(self) -> float:
        """Horizontal center, 0 = left edge, 1 = right edge."""
        return (self.x1 + self.x2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height as a fraction of frame height (a rough closeness cue)."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area as a fraction of frame area."""
        return self.width * self.height
