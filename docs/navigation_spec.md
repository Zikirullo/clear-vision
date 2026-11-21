# Navigation rules: specification

How Clear Vision turns one camera frame's detections into at most one short spoken
warning. This is the reference for any reimplementation (the phone app). The Python
code is `src/navigator.py` (pure functions) and `src/planner.py` (per-session state).

**The numbers are not repeated here as the source of truth.** The phone app loads them
from `models/nav_config.json`, which is generated from `configs/default.yaml` by
`scripts/export_nav_config.py`. Values quoted below are the current ones, for reading.

**Conformance:** `tests/golden/nav_cases.json` holds 35 input sequences with the exact
messages Python produces. An implementation conforms when it reproduces all of them:
same frames, texts, priorities and cooldown keys. Python checks itself against that file
in `tests/test_nav_golden.py`.

---

## 1. Inputs and state

Per frame:

| input | type | meaning |
|---|---|---|
| `t` | float, seconds | monotonic time of the frame (only differences matter) |
| `detections` | list | each: `class` (name from `classes`), `conf` (0-1), `box` = `[x1, y1, x2, y2]` |

Boxes are fractions of the frame: x from 0 (left edge) to 1 (right edge), y from 0 (top)
to 1 (bottom). The list order matters: it breaks ties (see 5 and 6).

State kept across frames (one session):

| state | meaning |
|---|---|
| `history` | the **candidates** (see 3a) of the last `max(window_frames, name_history_frames) - 1` frames (currently 4), oldest first |
| `last_spoken` | map cooldown key → time the message with that key was last spoken |
| `mute_non_hazards` | the hazards-only switch (starts at the config value; key `m` toggles it in the app) |

## 2. Geometry

| quantity | formula |
|---|---|
| `center_x` | `(x1 + x2) / 2` |
| `center_y` | `(y1 + y2) / 2` |
| `width`, `height` | `x2 - x1`, `y2 - y1` |
| `area` | `width * height` |
| `IoU(a, b)` | intersection area / (`area_a + area_b - intersection`); 0 if the union is 0. Intersection = `max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))` |

Use exactly these formulas, in this order of operations. Golden case
`boundary_center_equals_left_boundary` depends on the floating-point result of
`(0.233 + 0.433) / 2`.

**Direction (the corridor).** The frame is split into three vertical strips by
`center_x`. The middle strip is the walking corridor, "ahead". It is the full frame
height; there is no other corridor shape.

| condition | direction |
|---|---|
| `center_x < left_boundary` (0.333) | `left` |
| `center_x > right_boundary` (0.667) | `right` |
| otherwise (boundaries inclusive) | `ahead` |

**Proximity** (box height is the distance cue; no depth):

| condition | proximity |
|---|---|
| `height > near_height` (0.5) | `near` |
| `height >= mid_height` (0.25) | `mid` |
| otherwise | `far` |

## 3. Per-frame pipeline

For each frame, in this order:

**a. Candidates:** detections with `conf >= speak_conf[class]`, or `speak_conf_default`
(0.35) for classes not listed. Current values: car/bus/truck/motorcycle 0.45, person
0.35, pole/bollard/traffic_cone/fire_hydrant 0.25, curb/stairs/pothole/manhole 0.20.
Weaker detections are ignored for everything below. The live app still draws them gray.

**b. Speakable:** candidates that pass all three:
1. **Speaking zone** (near enough to matter):
   - vehicles (`vehicles`: car, bus, truck, motorcycle): `height > vehicle_min_height` (0.20)
   - all others: `y2 >= horizon_bottom` (0.4) **or** `height > min_speak_height` (0.15)
2. **Hazards only:** if `mute_non_hazards` is on, the class must be in `hazard_classes`.
3. **Confirmed** (section 4).

**c. Choose the message** from the speakable detections (section 5). The candidates of
this frame and the history are the context for the naming rule (section 6).

**d. Update history:** append this frame's candidates, and keep only the last
`max(window_frames, name_history_frames) - 1` (4) frames.

**e. If a message was chosen and spoken**, set `last_spoken[message.key] = t`. The golden
cases count every chosen message as spoken (see section 8 for the live app).

## 4. Confirmation (3 of 5 frames, same place)

`same_object(a, b)` is true when all hold:
- **same group:** `group(a.class) == group(b.class)`, where `group` is the name group
  (section 6) or the class itself, so a cone and a fire hydrant count as the same object
- **close centers:** `sqrt((a.center_x - b.center_x)² + (a.center_y - b.center_y)²) <= confirm_center_dist` (0.15)
- **overlap:** `IoU(a, b) >= confirm_iou` (0.3)

`track(det, n)`: the object's boxes over the last `n` frames, current one included:
```
track = [det]; last = det
for frame in the last (n - 1) history frames, newest first:
    match = the first detection d in frame (list order) with same_object(last, d)
    if match: track.append(match); last = match   # a moving object stays one object
```
Frames without a match don't break the chain; they just don't count.

`is_confirmed(det)` = `len(track(det, window_frames)) >= confirm_frames` (3 of 5).

## 5. Choosing the message (at most one per frame)

**Risk:** `risk = proximity_weights[proximity] * (ahead_weight if ahead else side_weight) * severity`.
- `proximity_weights`: near 1.0, mid 0.6, far 0.3
- `ahead_weight` 1.5, `side_weight` 1.0
- `severity[class]`, else `default_severity` (1). Current values: vehicles, bicycle,
  scooter, stairs and pothole 3; person, dog, the post group, barrier, curb and manhole 2;
  bench, chair, potted_plant, stop_sign, traffic_light and crosswalk 1.

```
sort speakable by risk, descending, stable (equal risks keep list order)
for det in that order:
    if risk(det) < risk_threshold (1.0): stop (no message)
    priority = "urgent" if direction == "ahead" and proximity == "near" else "normal"
    key = group(det.class) + ":" + direction          # e.g. "post:left", "car:ahead"
    wait = urgent_cooldown (2.0 s) if urgent else cooldown (4.0 s)
    if key in last_spoken and t - last_spoken[key] < wait: continue   # next riskiest object
    return Message(text, priority, key)       # text: section 7, name per section 6
```
The cooldown key ignores proximity and the spoken name: "Pole, right", then "Obstacle,
right, close" for the same post on the same side within 4 s is not repeated.

## 6. Naming rule (name groups)

The v1 model sometimes calls traffic cones and bollards "fire hydrant" (2026-11-18).
Classes that get confused form a **name group** (`name_groups`). Currently:
`post: [traffic_cone, bollard, fire_hydrant, pole]`.

A group member is spoken as `phrases.unknown` ("Obstacle") when **any** of these holds:
1. its `conf < name_min_conf` (0.6); exactly 0.6 keeps the name
2. another **candidate of the same frame** with a **different class of the same group**
   overlaps it with `IoU >= name_overlap_iou` (0.5)
3. **the class changed recently:** the boxes of `track(det, name_history_frames)`
   (section 4; currently 5 frames, the current one included) carry more than one class
   name, even if the current confidence is >= 0.6. Only candidates count, so a weak
   detection below its speak threshold doesn't. A different post elsewhere in the frame
   isn't the same object and doesn't count either. `name_history_frames: 1` turns this off.

**The warning itself is never dropped because of a name doubt.** Direction, distance,
priority and cooldown key stay the same; only the name changes. The group also counts
as one object for confirmation (section 4) and cooldowns (section 5).

Example: cone, hydrant, cone on one object at 0.8 is confirmed as one post and spoken as
"Obstacle ahead" (golden case `post_flicker_between_classes_obstacle`). A hydrant frame 4
frames back still makes the warning "Obstacle"; 5 frames back it is forgotten (golden cases
`post_class_change_four_frames_back_obstacle` and `..._five_frames_back_forgotten`).

Classes outside every group always use their own name.

## 7. Phrases

The spoken name of a class is `spoken_names[class]` if set (currently none). Otherwise
it is the class name with `_` replaced by a space, first character upper-case and the
rest lower-case: `traffic_cone` → "Traffic cone". `models/nav_config.json` lists the
result for every class under `spoken_names_resolved`, so the app doesn't need this rule.

| case | template (`phrases.*`) | example |
|---|---|---|
| ahead | `ahead`: `"{name} ahead"` | "Stairs ahead" |
| left / right | `side`: `"{name}, {direction}"` with `{direction}` = `directions.left` / `directions.right` | "Traffic cone, left" |
| near | append `close_suffix`: `", close"` | "Car ahead, close" |
| name doubt | `{name}` = `unknown`: `"Obstacle"` | "Obstacle, right, close" |

System messages (`phrases.system`), spoken by the app outside the rules above:

| key | text | priority | when |
|---|---|---|---|
| `started` | Clear Vision started | normal | after the model loaded |
| `camera_unavailable` | Camera not available | urgent | the camera can't be opened at start |
| `camera_lost` | Camera lost | urgent | no new frame for `camera.stale_timeout` (1.5 s); shorter gaps are ridden out silently |
| `camera_back` | Camera back | normal | once, when frames return |
| `model_failed` | Detection model failed. Clear Vision cannot run. | urgent | no model could load |
| `hazard_model_failed` | Hazard detector failed. Detecting people and vehicles only. | urgent | the hazard model failed, COCO fallback |
| `hazards_only` / `all_objects` | Hazards only / All objects | normal | the hazards-only switch was toggled |

## 8. What the live app adds (not in the golden cases)

- **Speech queue:** an urgent message interrupts current speech. A normal message is
  only spoken when nothing is being said, and only then counts as spoken (`last_spoken`
  is set), so it can come again on a later frame.
- **Detection thresholds come before all of this.** The model's per-class minimum
  confidence (`nav_config.json` → `detection.class_conf`: 0.15 for curb, stairs,
  pothole, pole and manhole, 0.25 otherwise) decides what becomes a detection at all.
  The speak thresholds in 3a are higher.

## 9. Config keys (`navigator:` in `configs/default.yaml`, `navigator` in `nav_config.json`)

| key | current | used in |
|---|---|---|
| `left_boundary`, `right_boundary` | 0.333, 0.667 | 2 direction |
| `near_height`, `mid_height` | 0.5, 0.25 | 2 proximity |
| `proximity_weights` | near 1.0, mid 0.6, far 0.3 | 5 risk |
| `ahead_weight`, `side_weight` | 1.5, 1.0 | 5 risk |
| `severity`, `default_severity` | per class, 1 | 5 risk |
| `risk_threshold` | 1.0 | 5 |
| `cooldown`, `urgent_cooldown` | 4.0 s, 2.0 s | 5 |
| `confirm_frames`, `window_frames` | 3, 5 | 4 |
| `confirm_center_dist`, `confirm_iou` | 0.15, 0.3 | 4 |
| `speak_conf`, `speak_conf_default` | per class, 0.35 | 3a |
| `horizon_bottom`, `min_speak_height` | 0.4, 0.15 | 3b zone |
| `vehicles`, `vehicle_min_height` | car, bus, truck, motorcycle; 0.20 | 3b zone |
| `mute_non_hazards`, `hazard_classes` | false; 14 classes | 3b |
| `name_groups`, `name_min_conf`, `name_overlap_iou` | post group; 0.6; 0.5 | 6 |
| `name_history_frames` | 5 | 6 (and history length, 1) |
| `phrases`, `spoken_names` | templates; {} | 7 |

## 10. Changing a rule

1. Change `src/navigator.py` and/or `configs/default.yaml`, and this document.
2. Run `python scripts/export_nav_config.py` (writes `models/nav_config.json`).
3. Run `python scripts/make_nav_golden.py` (rewrites the expected messages) and review the
   diff: every changed message must be intended.
4. `pytest` (`tests/test_nav_golden.py`), then update the phone app until its golden test
   passes too.

## 11. Known limits

- **No tracker yet:** objects are matched frame to frame only by class group, center
  distance and IoU (section 4).
- **Distance comes from box height,** which suits upright objects; flat hazards (curb,
  pothole, manhole, crosswalk) look "far" until close.
- **The class-change check follows one object** with the same matching as confirmation
  (section 4), so two posts that touch can be mixed up and spoken as "Obstacle". That
  errs on the side of a warning without a name.
