"""Navigation logic: turn detections into at most one short spoken warning.

Everything here is a pure function of its inputs (detections, config, clock
value, speech history), with no OpenCV or audio, so it is unit-tested without
a camera. Thresholds come from the `navigator:` section of configs/default.yaml.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from src.detection import Detection

Direction = Literal["left", "ahead", "right"]
Proximity = Literal["near", "mid", "far"]
Priority = Literal["urgent", "normal"]


@dataclass(frozen=True)
class NavigatorConfig:
    left_boundary: float = 0.333
    right_boundary: float = 0.667
    near_height: float = 0.5
    mid_height: float = 0.25
    proximity_weights: Mapping[str, float] = field(
        default_factory=lambda: {"near": 1.0, "mid": 0.6, "far": 0.3}
    )
    ahead_weight: float = 1.5
    side_weight: float = 1.0
    severity: Mapping[str, float] = field(default_factory=dict)
    default_severity: float = 1.0
    risk_threshold: float = 1.0
    cooldown: float = 4.0
    urgent_cooldown: float = 1.5

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> NavigatorConfig:
        """Build from the `navigator:` config section; unknown keys are ignored."""
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in cfg.items() if k in known})


DEFAULT_CONFIG = NavigatorConfig()


@dataclass(frozen=True)
class Message:
    """A warning to speak. `key` identifies it for cooldowns ("car:ahead")."""

    text: str
    priority: Priority
    key: str


def direction(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> Direction:
    """Which third of the frame the object's center is in."""
    if det.center_x < cfg.left_boundary:
        return "left"
    if det.center_x > cfg.right_boundary:
        return "right"
    return "ahead"


def proximity(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> Proximity:
    """Rough distance from box height: taller box = closer object."""
    if det.height > cfg.near_height:
        return "near"
    if det.height >= cfg.mid_height:
        return "mid"
    return "far"


def severity(class_name: str, cfg: NavigatorConfig = DEFAULT_CONFIG) -> float:
    """How dangerous a collision with this class would be."""
    return float(cfg.severity.get(class_name, cfg.default_severity))


def risk(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> float:
    """Combine proximity, being straight ahead, and class severity."""
    prox_weight = cfg.proximity_weights[proximity(det, cfg)]
    dir_weight = cfg.ahead_weight if direction(det, cfg) == "ahead" else cfg.side_weight
    return prox_weight * dir_weight * severity(det.class_name, cfg)


def message_key(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> str:
    return f"{det.class_name}:{direction(det, cfg)}"


def build_phrase(class_name: str, where: Direction, how_close: Proximity) -> str:
    """Short phrase such as "Car ahead, close" or "Person, left"."""
    name = class_name.capitalize()
    text = f"{name} ahead" if where == "ahead" else f"{name}, {where}"
    if how_close == "near":
        text += ", close"
    return text


def is_on_cooldown(
    key: str,
    priority: Priority,
    now: float,
    last_spoken: Mapping[str, float],
    cfg: NavigatorConfig = DEFAULT_CONFIG,
) -> bool:
    """True if `key` was spoken too recently to repeat.

    Urgent warnings use the shorter `urgent_cooldown`, so an obstacle that keeps
    approaching straight ahead is repeated more often.
    """
    last = last_spoken.get(key)
    if last is None:
        return False
    wait = cfg.urgent_cooldown if priority == "urgent" else cfg.cooldown
    return now - last < wait


def choose_message(
    detections: Iterable[Detection],
    now: float,
    last_spoken: Mapping[str, float],
    cfg: NavigatorConfig = DEFAULT_CONFIG,
) -> Message | None:
    """Pick the single most dangerous object worth announcing, if any.

    Objects below `risk_threshold` are ignored. If the riskiest object is on
    cooldown, the next riskiest one is considered, so a parked car that was
    already announced doesn't hide a new person walking toward the user.

    Pure: `last_spoken` (key -> time last spoken) is not modified. After the
    message is actually spoken, the caller records `last_spoken[msg.key] = now`.
    """
    scored = sorted(
        ((risk(d, cfg), d) for d in detections),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for score, det in scored:
        if score < cfg.risk_threshold:
            break
        where = direction(det, cfg)
        how_close = proximity(det, cfg)
        priority: Priority = "urgent" if where == "ahead" and how_close == "near" else "normal"
        key = message_key(det, cfg)
        if is_on_cooldown(key, priority, now, last_spoken, cfg):
            continue
        return Message(text=build_phrase(det.class_name, where, how_close), priority=priority, key=key)
    return None
