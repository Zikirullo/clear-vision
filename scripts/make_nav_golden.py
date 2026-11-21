"""Generate tests/golden/nav_cases.json: navigation input cases with the messages the Python
implementation produces. The phone app's tests must reproduce them exactly.

    python scripts/make_nav_golden.py      # only after an intended change to the rules

Each case is a sequence of frames (time in seconds + detections with normalized boxes).
Frames are replayed through src/planner.py SpeechPlanner with the numbers from
models/nav_config.json; every planned message counts as spoken at its frame time (as
with `--mute`; the live app may hold back normal messages while speech is playing).
Box numbers are rounded to 4 decimals before the expected output is computed, so the
stored inputs are exactly what produced the stored outputs.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection import Detection  # noqa: E402
from src.navigator import NavigatorConfig  # noqa: E402
from src.planner import SpeechPlanner  # noqa: E402

NAV_CONFIG = ROOT / "models" / "nav_config.json"
OUT = ROOT / "tests" / "golden" / "nav_cases.json"
FPS = 10.0


def det(cls: str, conf: float, cx: float = 0.5, top: float = 0.4, h: float = 0.5, w: float = 0.2) -> dict[str, Any]:
    """Detection centered at `cx`, from `top` down `h` tall (all fractions of the frame)."""
    box = [cx - w / 2, top, cx + w / 2, top + h]
    return {"class": cls, "conf": conf, "box": [round(v, 4) for v in box]}


def frames(per_frame: Sequence[Sequence[dict[str, Any]]], start: float = 0.0, fps: float = FPS) -> list[dict[str, Any]]:
    return [{"t": round(start + i / fps, 4), "detections": list(d)} for i, d in enumerate(per_frame)]


def repeat(dets: Sequence[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    return [list(dets) for _ in range(n)]


def cases() -> list[dict[str, Any]]:
    stairs_close = det("stairs", 0.8, cx=0.5, top=0.4, h=0.55)
    stairs_left_mid = det("stairs", 0.7, cx=0.2, top=0.6, h=0.3)
    car_close = det("car", 0.9, cx=0.5, top=0.35, h=0.6)
    cone = det("traffic_cone", 0.8, cx=0.5, top=0.5, h=0.3)
    c: list[dict[str, Any]] = []

    def add(cid: str, desc: str, fr: list[dict[str, Any]], mute: bool = False) -> None:
        c.append({"id": cid, "description": desc, "mute_non_hazards": mute, "frames": fr})

    # --- confirmation
    add("steady_stairs_close_ahead", "Steady stairs ahead and close: spoken once on the 3rd frame, urgent.",
        frames(repeat([stairs_close], 10)))
    add("flicker_one_of_five_silent", "A car in 1 of every 5 frames never confirms.",
        frames([[car_close] if i % 5 == 0 else [] for i in range(25)]))
    add("two_of_five_silent", "Seen twice in five frames: not confirmed.",
        frames([[stairs_close], [], [stairs_close], [], [], []]))
    add("three_of_five_with_gaps", "Seen 3 times within 5 frames, with gaps: spoken on the 3rd sighting.",
        frames([[stairs_close], [], [stairs_close], [], [stairs_close]]))
    add("slow_motion_same_object", "An object moving 2% of the frame per frame stays one object.",
        frames([[det("person", 0.8, cx=0.40 + 0.02 * i, w=0.15)] for i in range(6)]))
    add("jumping_boxes_not_confirmed", "Same class in a different place every frame (walls as poles): silent.",
        frames([[det("pole", 0.9, cx=x, w=0.06)] for x in (0.1, 0.5, 0.9, 0.3, 0.7, 0.1, 0.5)]))
    # --- cooldowns
    add("normal_repeat_after_4s", "Normal warning repeats after the 4 s cooldown.",
        frames(repeat([stairs_left_mid], 50)))
    add("urgent_repeat_after_2s", "Urgent (ahead + close) warning repeats after 2 s.",
        frames(repeat([stairs_close], 50)))
    add("cooldown_falls_through_to_next_object", "While the car is on cooldown, the next riskiest object is spoken.",
        frames(repeat([car_close, det("person", 0.8, cx=0.15, top=0.5, h=0.3)], 12)))
    # --- direction and proximity
    add("direction_left", "Center left of 0.333: left.", frames(repeat([det("person", 0.8, cx=0.2, top=0.5, h=0.3)], 4)))
    add("direction_right", "Center right of 0.667: right.", frames(repeat([det("person", 0.8, cx=0.8, top=0.5, h=0.3)], 4)))
    add("boundary_center_equals_left_boundary", "center_x = (x1 + x2) / 2 compared with < left_boundary.",
        frames(repeat([{"class": "person", "conf": 0.8, "box": [0.233, 0.5, 0.433, 0.8]}], 4)))
    add("far_side_person_below_risk", "Far person to the side: risk 0.6 < 1, silent.",
        frames(repeat([det("person", 0.9, cx=0.15, top=0.6, h=0.2)], 6)))
    add("mid_ahead_not_close", "Mid distance ahead: normal priority, no 'close'.",
        frames(repeat([det("stairs", 0.8, cx=0.5, top=0.6, h=0.3)], 4)))
    # --- speak thresholds
    add("car_below_speak_threshold", "Car at 0.44 < 0.45: silent.", frames(repeat([det("car", 0.44, h=0.6, top=0.35)], 6)))
    add("car_above_speak_threshold", "Car at 0.46: spoken.", frames(repeat([det("car", 0.46, h=0.6, top=0.35)], 4)))
    add("curb_threshold_two_objects", "Curb 0.19 (left) silent, curb 0.21 (right) spoken.",
        frames(repeat([det("curb", 0.19, cx=0.2, top=0.7, h=0.25, w=0.3),
                       det("curb", 0.21, cx=0.8, top=0.7, h=0.25, w=0.3)], 4)))
    # --- speaking zone
    add("small_far_pole_near_horizon_silent", "Pole whose bottom is above 0.4 and 10% tall: too far, silent.",
        frames(repeat([det("pole", 0.9, top=0.2, h=0.1, w=0.03)], 6)))
    add("tall_object_high_in_frame_spoken", "Bottom above 0.4 but taller than 15%: spoken.",
        frames(repeat([det("pole", 0.9, cx=0.8, top=0.05, h=0.3, w=0.05)], 4)))
    add("vehicle_needs_twenty_percent", "Car 18% tall (low in frame) silent until it grows to 30%.",
        frames(repeat([det("car", 0.9, top=0.7, h=0.18)], 4) + repeat([det("car", 0.9, top=0.6, h=0.3)], 4)))
    # --- hazards only
    add("hazards_only_mutes_dog", "Hazards-only: the dog is silent, the stairs are spoken.",
        frames(repeat([det("dog", 0.9, cx=0.2, top=0.5, h=0.4), det("stairs", 0.9, cx=0.8, top=0.5, h=0.4)], 4)), mute=True)
    add("all_objects_speaks_dog", "Hazards-only off: the dog is spoken.",
        frames(repeat([det("dog", 0.9, cx=0.2, top=0.5, h=0.4)], 4)))
    add("bench_side_low_risk_silent", "Bench (severity 1) at mid distance to the side: risk 0.6, silent.",
        frames(repeat([det("bench", 0.9, cx=0.2, top=0.6, h=0.3)], 6)))
    # --- naming rule for the post group
    add("post_confident_name_spoken", "Traffic cone at 0.8: its name is spoken.", frames(repeat([cone], 4)))
    add("post_low_confidence_obstacle", "Traffic cone at 0.55 < 0.6: spoken as Obstacle, warning kept.",
        frames(repeat([det("traffic_cone", 0.55, cx=0.2, top=0.5, h=0.3)], 4)))
    add("post_name_at_exactly_min_conf", "Pole at exactly 0.6: name kept (rule is < 0.6).",
        frames(repeat([det("pole", 0.6, cx=0.8, top=0.5, h=0.4, w=0.05)], 4)))
    add("post_two_classes_overlap_obstacle", "Cone 0.8 and fire hydrant 0.75 on one object: Obstacle.",
        frames(repeat([cone, det("fire_hydrant", 0.75, cx=0.51, top=0.5, h=0.3)], 4)))
    add("post_overlap_with_weak_detection_ignored", "A bollard at 0.2 (below its 0.25 speak threshold) doesn't make the pole doubtful.",
        frames(repeat([det("pole", 0.8, cx=0.5, top=0.4, h=0.5, w=0.06), det("bollard", 0.2, cx=0.5, top=0.4, h=0.5, w=0.06)], 4)))
    add("post_flicker_between_classes_obstacle", "Cone/hydrant alternate on one object at 0.8: confirmed as one post, "
        "spoken once as Obstacle (the class changed within the last 5 frames).",
        frames([[det("traffic_cone" if i % 2 == 0 else "fire_hydrant", 0.8, top=0.5, h=0.3)] for i in range(12)]))
    add("post_group_shares_cooldown", "Pole spoken, then the same object at 0.5 (would be Obstacle): same key post:right, no repeat.",
        frames(repeat([det("pole", 0.8, cx=0.8, top=0.5, h=0.4, w=0.05)], 5)
               + repeat([det("pole", 0.5, cx=0.8, top=0.5, h=0.4, w=0.05)], 15)))
    # --- naming rule: class changes over the last name_history_frames (5) frames
    hydrant = det("fire_hydrant", 0.8, cx=0.5, top=0.5, h=0.3)
    add("post_class_change_four_frames_back_obstacle", "Steady cone, one hydrant frame 4 frames before the cooldown "
        "ends (inside the 5-frame window): the repeat is Obstacle.",
        frames([[hydrant] if i == 38 else [cone] for i in range(50)]))
    add("post_class_change_five_frames_back_forgotten", "Same, but the hydrant frame is 5 frames back (outside the "
        "window): the repeat keeps the name.",
        frames([[hydrant] if i == 37 else [cone] for i in range(50)]))
    add("post_two_different_posts_keep_names", "A pole on the left and a bollard on the right are different objects: "
        "both keep their names.",
        frames(repeat([det("pole", 0.8, cx=0.2, top=0.5, h=0.4, w=0.05),
                       det("bollard", 0.8, cx=0.8, top=0.5, h=0.4, w=0.05)], 6)))
    add("post_weak_other_class_in_history_ignored", "A hydrant at 0.2 (below its speak threshold) before a steady cone "
        "doesn't count as a class change.",
        frames([[det("fire_hydrant", 0.2, cx=0.5, top=0.5, h=0.3)]] + repeat([cone], 5)))
    add("post_same_class_low_then_high_conf_keeps_name", "A pole at 0.5 then 0.8: same class name, so only the "
        "current confidence matters.",
        frames(repeat([det("pole", 0.5, cx=0.8, top=0.5, h=0.4, w=0.05)], 2)
               + repeat([det("pole", 0.8, cx=0.8, top=0.5, h=0.4, w=0.05)], 2)))
    return c


def run_case(case: dict[str, Any], nav: NavigatorConfig) -> list[dict[str, Any]]:
    planner = SpeechPlanner(nav)
    planner.mute_non_hazards = case["mute_non_hazards"]
    out = []
    for i, frame in enumerate(case["frames"]):
        dets = [Detection(d["class"], d["conf"], *d["box"]) for d in frame["detections"]]
        plan = planner.update(dets, frame["t"])
        if plan.message is not None:
            planner.mark_spoken(plan.message, frame["t"])
            out.append({"frame": i, "t": frame["t"], "text": plan.message.text,
                        "priority": plan.message.priority, "key": plan.message.key})
    return out


def load_nav(path: Path = NAV_CONFIG) -> NavigatorConfig:
    return NavigatorConfig.from_dict(json.loads(path.read_text())["navigator"])


def main() -> int:
    nav = load_nav()
    built = cases()
    for case in built:
        case["expected"] = run_case(case, nav)
    doc = {
        "format_version": 1,
        "config": "models/nav_config.json",
        "spec": "docs/navigation_spec.md",
        "semantics": ("Replay each case's frames in order with a fresh state. A frame's message (if any) "
                      "counts as spoken at that frame's time. Box = [x1, y1, x2, y2], fractions of the frame. "
                      "Expected = every message, with frame index, time, text, priority and cooldown key."),
        "cases": built,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(built)} cases, "
          f"{sum(len(c['expected']) for c in built)} expected messages, "
          f"{sum(1 for c in built if not c['expected'])} silent cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
