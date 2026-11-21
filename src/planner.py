"""Per-session speech state around the pure navigator functions.

Holds the last few frames' candidate detections (for confirmation and the naming
rule's class-change check), when each
message was last spoken (for cooldowns) and the hazards-only toggle. Used by
main.py and scripts/replay.py so both decide exactly the same way.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from src.detection import Detection
from src.navigator import Message, NavigatorConfig, SpeechPlan, plan_speech


class SpeechPlanner:
    def __init__(self, cfg: NavigatorConfig) -> None:
        self.cfg = cfg
        self.mute_non_hazards = cfg.mute_non_hazards
        self.history: deque[tuple[Detection, ...]] = deque(
            maxlen=max(1, cfg.window_frames - 1, cfg.name_history_frames - 1)
        )
        self.last_spoken: dict[str, float] = {}

    def update(self, detections: Iterable[Detection], now: float) -> SpeechPlan:
        """Plan this frame and remember its candidates for the next frames' confirmation."""
        plan = plan_speech(detections, self.history, now, self.last_spoken, self.cfg, self.mute_non_hazards)
        self.history.append(plan.candidates)
        return plan

    def mark_spoken(self, msg: Message, now: float) -> None:
        self.last_spoken[msg.key] = now

    def toggle_mute_non_hazards(self) -> bool:
        self.mute_non_hazards = not self.mute_non_hazards
        return self.mute_non_hazards
