"""Navigation logic: turn detections into at most one short spoken warning.

Everything here is a pure function of its inputs (detections, config, clock
value, speech history), with no OpenCV or audio, so it is unit-tested without
a camera. Thresholds come from the `navigator:` section of configs/default.yaml.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    urgent_cooldown: float = 2.0
    # Speech filter: what is allowed to be spoken (detection thresholds stay separate).
    confirm_frames: int = 3  # seen in at least this many ...
    window_frames: int = 5  # ... of the last this many frames (current one included)
    confirm_center_dist: float = 0.15  # same place: centers at most this far apart (fraction of the frame)
    confirm_iou: float = 0.3  # ... and at least this IoU with the previous box
    speak_conf: Mapping[str, float] = field(default_factory=dict)
    speak_conf_default: float = 0.35
    horizon_bottom: float = 0.4  # box bottom at or below this (lower 60% of the frame) ...
    min_speak_height: float = 0.15  # ... or box taller than this; otherwise too far to mention
    vehicles: Sequence[str] = ("car", "bus", "truck", "motorcycle")
    vehicle_min_height: float = 0.20  # vehicles: spoken only when taller than this
    mute_non_hazards: bool = False
    hazard_classes: Sequence[str] = (
        "curb", "stairs", "pothole", "manhole", "pole", "bollard", "traffic_cone", "fire_hydrant",
        "barrier", "person", "car", "bus", "truck", "motorcycle",
    )

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
    """Short phrase such as "Car ahead, close" or "Traffic cone, left"."""
    name = class_name.replace("_", " ").capitalize()
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


# ---------------------------------------------------------------------- speech filter


def speak_threshold(class_name: str, cfg: NavigatorConfig = DEFAULT_CONFIG) -> float:
    """Confidence an object needs to be spoken (detection can use a lower one)."""
    return float(cfg.speak_conf.get(class_name, cfg.speak_conf_default))


def confident_enough(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> bool:
    return det.confidence >= speak_threshold(det.class_name, cfg)


def in_speaking_zone(det: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> bool:
    """Near enough to matter: vehicles by height; others by bottom edge in the lower
    part of the frame or by height (small objects near the horizon stay silent)."""
    if det.class_name in cfg.vehicles:
        return det.height > cfg.vehicle_min_height
    return det.y2 >= cfg.horizon_bottom or det.height > cfg.min_speak_height


def allowed_class(det: Detection, mute_non_hazards: bool, cfg: NavigatorConfig = DEFAULT_CONFIG) -> bool:
    return not mute_non_hazards or det.class_name in cfg.hazard_classes


def same_object(a: Detection, b: Detection, cfg: NavigatorConfig = DEFAULT_CONFIG) -> bool:
    """Same class, roughly the same place: centers close and boxes overlapping."""
    if a.class_name != b.class_name:
        return False
    ax, ay = a.center_x, (a.y1 + a.y2) / 2
    bx, by = b.center_x, (b.y1 + b.y2) / 2
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > cfg.confirm_center_dist:
        return False
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return union > 0 and inter / union >= cfg.confirm_iou


def is_confirmed(
    det: Detection, history: Sequence[Sequence[Detection]], cfg: NavigatorConfig = DEFAULT_CONFIG
) -> bool:
    """True if `det` (current frame) was also seen in enough of the previous frames.

    `history` holds earlier frames' candidate detections, oldest first. Walking back
    from the current frame, each frame may contribute one box that matches the last
    matched box (so a slowly moving object stays the same object). Frames without a
    match don't break the chain; they just don't count.
    """
    count, last = 1, det
    previous = list(history)[-(cfg.window_frames - 1):] if cfg.window_frames > 1 else []
    for frame in reversed(previous):
        match = next((d for d in frame if same_object(last, d, cfg)), None)
        if match is not None:
            count, last = count + 1, match
    return count >= cfg.confirm_frames


@dataclass(frozen=True)
class SpeechPlan:
    """What happened to this frame's detections, and the message to say (if any)."""

    candidates: tuple[Detection, ...]  # confident enough to be spoken
    speakable: tuple[Detection, ...]  # ... and confirmed, near enough, not muted
    message: Message | None


def plan_speech(
    detections: Iterable[Detection],
    history: Sequence[Sequence[Detection]],
    now: float,
    last_spoken: Mapping[str, float],
    cfg: NavigatorConfig = DEFAULT_CONFIG,
    mute_non_hazards: bool | None = None,
) -> SpeechPlan:
    """Filter one frame's detections for speech and pick at most one message.

    Pure: the caller appends `plan.candidates` to `history` afterwards and records
    `last_spoken[msg.key] = now` once the message is actually spoken.
    """
    mute = cfg.mute_non_hazards if mute_non_hazards is None else mute_non_hazards
    candidates = tuple(d for d in detections if confident_enough(d, cfg))
    speakable = tuple(
        d for d in candidates
        if in_speaking_zone(d, cfg) and allowed_class(d, mute, cfg) and is_confirmed(d, history, cfg)
    )
    return SpeechPlan(candidates, speakable, choose_message(speakable, now, last_spoken, cfg))
