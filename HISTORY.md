# History

What was done before each commit, newest first. Add a new entry before every commit
(see "History log" in CLAUDE.md).

---

## feat: VERIFY_ONLY in notebook 02 for an upload that stopped before merged.done (2026-11-25)

- Why: the v2 append stopped during verification. `merged.zip` (7.91 GB, uploaded
  2026-11-24 04:22 UTC) and `merge_report.json` were on Drive, but `merged.done` was
  missing, so training refuses the zip.
- `scripts/merge_datasets.py`:
  - `verify_existing(zip, work)`: flushes Drive, copies the zip to the local disk and
    compares sizes, then runs `testzip()`, the CRC of every member.
  - It checks `data.yaml` is present, and that the `merge_report.json` inside the zip
    equals the one on Drive.
  - It counts images and label files per split (train, val, test, external_test) and per
    source (file-name prefix, `negatives` included). It compares them with the report and
    flags images without labels, labels without images, unknown sources and a
    `phash_index.json` mismatch.
  - Writes `merged.done` (same fields, via the new `done_info()` that `publish_zip` now uses
    too) only when there are no problems; never overwrites an existing one. CLI:
    `--verify-only` (`--raw` not needed).
- Notebook 02: `VERIFY_ONLY = False` setting. When True it skips downloads, the FiftyOne
  install and the merge, verifies, and raises if it fails. The section 3 text explains how
  to go back to the previous zip with Drive's version history.
- Tested:
  - 6 new tests in `tests/test_merge.py`: the done content matches a normal run and
    `prepare_dataset` accepts it; a corrupt member, a truncated zip, report counts that
    differ and a missing image are each rejected with no done written; member counting.
  - Verify runs after the v2 real-photo appends: external_test and negatives counts.
  - `pytest`: 367 passed.
- Not tested: the real 7.91 GB zip (Colab only). Copy plus CRC test reads it twice, which
  is probably 10-20 minutes in Colab.

---

## feat: screen the COCO negatives with budget_v1 and top up to ~3,000 (2026-11-24)

- Why: a "negative" with unlabeled stairs, a curb or a pole would teach v2 to ignore real
  hazards. Indoor COCO scenes can contain them (COCO doesn't label stairs or curbs).
- `scripts/download_datasets.py`:
  - `NEGATIVE_HAZARDS` (stairs, curb, pole, pothole, manhole, bollard, barrier,
    traffic_cone), `SCREEN_CONF` 0.25, `SCREEN_IMGSZ` 512, and `hazard_hits()`.
  - `collect_negatives(batches, n, screen)` takes eligible images (same rules as
    `pick_negatives`) from growing candidate batches. Each new image is screened once,
    images with a hazard detection are dropped, and it stops at n clean images. It reports
    when there are not enough.
  - `make_yolo_screen()` runs an Ultralytics model. `draw_hits()` draws what it saw.
  - `download_negatives(..., screen=, screen_name=)` asks FiftyOne for 4x n candidates,
    then 8x, 16x ... until n remain or COCO returns no new images.
  - `.done` records `screened_with` and `dropped_by_screen`. A download done with another
    screen (or none) is redone.
- Notebook 02:
  - New settings: `NEGATIVES_SCREEN_MODEL` (exports/budget_v1/best.pt),
    `NEGATIVES_SCREEN_CONF` 0.25 and `NEGATIVES_SCREEN_IMGSZ` 512. It fails loudly if the
    model is missing.
  - Prints kept/dropped counts, dropped images per class, and a warning if fewer than
    wanted. Shows 8 dropped examples with budget_v1's boxes in red.
- Tested:
  - 6 new tests in `tests/test_download.py`:
    - hazard filter: classes and conf;
    - drop + top-up from the next batch, each image screened once, ineligible images never
      screened, stop when enough;
    - not enough; no screen;
    - `download_negatives` with a fake FiftyOne: candidate sizes, `.done` info, skip when
      done, redo with another screen, stop when COCO has no more;
    - drawing.
  - `pytest`: 361 passed.
  - On the Mac, `make_yolo_screen` with the real budget_v1 flagged hazards in street photos
    (pole, bollard, manhole).
- Not tested: the real FiftyOne download and screening in Colab. How many COCO indoor images
  budget_v1 flags (it over-detects poles and manholes) is unknown. If it flags many,
  topping up may need several larger FiftyOne downloads, which take longer.

---

## feat: v2 data: four Roboflow sources, focus-only append, pHash leakage guard, external test and negatives (2026-11-24)

- Why: budget_v1 is weakest on curb, stairs, pothole, pole and manhole. On Nov 18 it scored
  0.98 on cones in the test split but found no cone in 24 real street photos, because the
  test split comes from the same Roboflow sources as training. Indoors it said "car ahead".
- `configs/datasets.yaml`: rf_freesight (kaan-bisfl/freesight-akli2 v1), rf_neytra
  (jeet-xlgwq/neytra-v3-sxshi v4), rf_safewalk (safewalkbd-7jrsx/safewalkbd-final v31) and
  rf_drone_obstacles (droneobstacles-bvrmw/drone-obstacles-acuxy v4), all CC BY 4.0. The
  existing fallback uses the newest version if one fails.
- `configs/mappings/` for the four sources and `negatives`:
  - `unlisted: drop` and `missing: warn`, written before download, so the spellings are
    guesses. Notebook 02 prints each dataset's real classes and where each goes, plus
    `CHECK:` lines for dropped names that look like ours.
  - `Light` is dropped and saved as review samples (map to traffic_light after checking).
  - Missing mapped names are one warning line per source, not ~50.
- `scripts/merge_datasets.py` (append):
  - `focus_only` per source keeps groups with curb/stairs/pothole/pole/manhole boxes, plus
    other groups up to 20% of the kept images. Roboflow augmented copies stay together.
  - pHash leakage guard (imagehash, decoded at reduced size). The merged images' hashes are
    cached in `phash_index.json` inside the zip:
    - a new image within 6 bits of a val/test/external_test image is dropped;
    - one near a train image goes to train;
    - near-duplicates among new images are joined into one group (across sources: train);
    - a new external_test image near any other image is dropped.
    - Counts per source are printed and kept in the report (`focus_dropped`,
      `near_dup_dropped`, `near_dup_to_train`).
  - `split: external_test` in a mapping file holds the whole source out (rf_safewalk).
    `data.yaml` lists `external_test` only when the split exists. The report and class
    table get an external column only when there is data.
  - Notebook 02 resizes appended images to 800 px (`APPEND_MAX_SIDE`). The class table and
    zip size are printed before upload, and a failed space check stops before upload.
- `scripts/download_datasets.py`:
  - `class_table`, `suspicious_drops`, and a verbose `check_mappings`.
  - Negatives: `pick_negatives` (an indoor COCO class and none of person, bicycle, car,
    motorcycle, bus, truck, dog, bench, fire hydrant, traffic light, or outdoor cues),
    `write_negatives` (empty label files), and `download_negatives` (FiftyOne coco-2017
    train, 4x candidates, 3,000 picked).
- `scripts/dataset_report.py` + notebook 03:
  - Priority classes (curb, stairs, pothole, pole, manhole, traffic_cone, bollard) are
    repeated 2-6x even when common.
  - `NO_REPEAT_SOURCES = {traffic_cone: [rf_cone]}`: rf_cone is a general cone dataset
    (sports cones, product photos) and gets no extra cone repeats.
- Evaluation:
  - `compare_detectors.extract_test_split(..., splits=)` extracts external_test too.
  - `safety_report.py` + notebook 10 report `test`, `external_test` and the street photos
    separately. The JSON now has one section per data set.
  - The street photos have no boxes, so the check is per photo: class found anywhere, plus
    the other names given. Classes come from `tests/assets/street_photos/expected.yaml` (by
    file-name prefix).
- `tests/assets/street_photos/`: the 24 Wikimedia photos, resized to 800 px (3.1 MB), with
  README (licenses: per-file Commons licenses; attribution still to be added before sharing).
  Paths in make_photo_video, mobile README, CLAUDE.md and requirements-export updated.
- `requirements.txt`: imagehash.
- Tested:
  - `tests/test_merge_v2.py` (18): the focus filter (20% cap, groups kept together, only
    listed sources), and pHash on real photos (re-exports match, other photos don't).
  - Also: distances and the popcount fallback, both guards (drop, to train, joined groups,
    external), forced splits.
  - An end-to-end append on real photos: focus only, the copy of a test image dropped, the
    copy of a train image in train, pair kept together, external split and data.yaml, the
    pHash index. Plus an append of negatives with empty labels, and the real mapping/dataset
    entries.
  - Added tests for the class check, negatives selection and writing, priority and
    no-repeat balancing, the street check, and external_test extraction.
  - The old append tests turn the guard off: their synthetic fixtures all share one pHash.
  - `pytest`: 355 passed. `flutter test`: 55 passed.
  - Notebook 10's cells ran locally with budget_v1 on a fake zip with test + external_test.
    Notebook code compiles.
- budget_v1 street-photo baseline (Mac, conf 0.15): manhole 4/4, stairs 3/4, pothole 3/4,
  bollard 1/4, pole 1/4, traffic_cone 0/4, so 12 of 24 photos. Cones were called
  fire_hydrant/barrier, poles car/truck.
- Not tested / open:
  - Nothing was downloaded: the real class names of the four datasets and their versions
    are unchecked. Notebook 02 prints them; mappings may need fixes.
  - The FiftyOne negatives download wasn't run, and COCO "indoor" images may still contain
    unlabeled stairs.
  - Zip size after the append is unknown; the space check guards the upload.
  - Leakage pHash timing on the full dataset (~tens of thousands of images) wasn't measured.
  - Whether rf_cone really is mostly sports cones was not checked by eye (notebook 03's
    grid shows samples).
  - Notebook 09 (pseudo-labels) leaves external_test alone; re-run it with
    `FORCE_PSEUDO = True` after the append, as before.

---

## feat: safety report (recall by object size against the proposed safety target) (2026-11-24)

- Why: mAP doesn't say whether a model is safe to walk with. The proposed target: for each
  hazard class (curb, stairs, pothole, pole, manhole, bollard, barrier, traffic_cone,
  scooter, crosswalk) and person, find at least 90% of the medium and large objects.
- `scripts/safety_report.py` (pure functions + CLI):
  - Matching: same class, IoU >= 0.5, each ground-truth box found once, highest
    confidence first (as notebook 08's mAP).
  - Size groups by box area share: small < 1% (far), medium 1-5%, large > 5% (close).
  - Recall per group + medium+large, precision per class, PASS if medium+large recall
    >= 0.90 ("no data" when a class has no medium/large boxes).
  - Runs the model once at the lowest threshold (0.15) and drops weaker predictions for
    0.25. Same boxes as a second run, since NMS only lets a stronger box remove a weaker one.
  - Test split from merged.zip with notebook 08's `extract_test_split`; imgsz from
    `metrics.json` next to the model (`mobile_export.read_imgsz`).
  - JSON to `runs/safety_report/<model>_<date>.json` (model = run folder for best.pt).
    It includes every medium/large miss.
  - `draw_miss`: missed box red, the model's predictions yellow with names. That shows
    hazards found as the wrong class.
- `configs/default.yaml`: new `safety_target` section (classes, conf levels 0.15/0.25,
  IoU, size limits, 0.90).
- `notebooks/10_safety_report.ipynb`: header like notebooks 02/07, the fixed Kaggle check
  (env vars, asserted against `is_kaggle()`), all imports in Setup, T4 metadata. Settings:
  `MODEL_PATH` (default `exports/budget_v1/best.pt`). One table per threshold, saves the
  JSON, then 8 missed medium/large examples per failing class (at 0.15).
- Tested:
  - `tests/test_safety_report.py` (16) uses a fake split and fake predictions. It covers size
    boundaries, the IoU cut-off, duplicates as false positives, best-overlap matching,
    recall/precision at both thresholds, misses only medium/large, no-data, PASS at exactly
    0.90, small misses not failing a class, the table, config values, the JSON name and
    content, label files, example picking and drawing.
  - The tests caught `pick_misses` collapsing equal misses; fixed.
  - `pytest`: 328 passed. `flutter test`: 55 passed (unchanged).
  - Smoke test on the Mac: the CLI and the notebook's cells 2-6 ran with the real budget_v1
    model on 3 street photos with fake labels (imgsz 512 read from metrics.json). The plot
    was checked by eye.
- Not tested: a real run on the merged test split (Colab/Drive only), so budget_v1's
  actual numbers are not known yet. Not run on Kaggle. Precision is a lower bound because
  sources label only their own classes.

---

## feat: Flutter phone app (iPhone first, Android ready) with the Dart port of the rules (2026-11-21)

- Why: run Clear Vision on the user's iPhone with the same rules as the Mac app.
- `mobile_flutter/` (Flutter 3.50, Dart 3.14), org `com.clearvision`:
  - Detection: `ultralytics_yolo` 0.7.2 `YOLOView`, model found by type in `assets/models/`
    (`*.mlpackage.zip` on iOS, `*.tflite` on Android), task detect, plugin threshold =
    lowest `detection.class_conf` (0.15), then per-class thresholds in Dart. `onResult`
    feeds our rules; the plugin draws boxes only in the debug view.
  - `lib/navigation/`: Dart port of `src/navigator.py` + `src/planner.py` (confirmation,
    speak thresholds, speaking zone, cooldowns, post naming rule incl. class-change window),
    plus pure health checks and vibration patterns. Stable sort by risk (Dart's sort isn't
    stable); `pow(x, 0.5)` like Python's `** 0.5`.
  - Speech: flutter_tts, urgent interrupts, normal only when quiet (counts as spoken only
    then); iOS audio category `playback` (plays on silent, ducks music); Android navigation
    audio attributes + focus. Vibration: pulses = side, timing = distance.
  - UI: one full-screen high-contrast button (double-tap start/stop, long press = describe
    scene, single tap = status, triple-tap = debug view), VoiceOver labels, hints and rotor
    actions, settings icon; settings for voice speed, language (en; ko/uz packs later),
    normal/silent, vibration, hazards only; wakelock while running; auto-start on load.
  - Health: camera lost after `camera.stale_timeout` (1.5 s) + "Camera back" once; slow
    below 8 FPS for 3 s (repeat <= every 60 s); battery 20% / 10%; model failure and
    label mismatch spoken.
  - iOS: iOS 15 minimum (Podfile), portrait only, NSCameraUsageDescription. Android:
    CAMERA + VIBRATE permissions, TTS query, portrait, minSdk 24 / compileSdk 36.
  - README: Mac setup, signing with a free Apple ID, Developer Mode, 7-day limit,
    TestFlight needs the paid program, Android setup + APK, license (AGPL-3.0).
- Config: new `mobile:` section in `configs/default.yaml`; new phone system phrases
  (stopped, nothing_detected, slow, battery_low/critical, labels_mismatch).
  `scripts/export_nav_config.py` now writes an `app` block (mobile + stale_timeout);
  golden cases unchanged (35, identical outputs).
- `scripts/sync_mobile_assets.py`: copies nav_config.json, labels.txt and a run's models
  into the app (`--run v2` swaps the model, older files removed). Models gitignored.
- `scripts/make_photo_video.py`: slideshow video from photos (the 48 s street video).
- Notebook 07: no change needed. The plugin reads task/names from the embedded metadata;
  checked our `.mlpackage` (task detect, 20 names, imgsz 512, NMS pipeline) against
  UltralyticsYOLO 8.9.23 (handles NMS-pipeline models) and the Android decoder (own NMS,
  normalized or pixel boxes).
- Docs: spec section 8 (phone extras, new system messages), CLAUDE.md (phone workflow).
- Tests:
  - `flutter test` 55 passed: all 35 golden cases read from `tests/golden/`; checked that
    making the left boundary inclusive fails exactly the boundary case. Also health,
    vibration, scene description, config, planner, tap counting. `flutter analyze` clean.
  - `pytest` 312 passed (7 new: sync script, app config copy, labels, photo video).
  - `flutter build ios --release --no-codesign` built Runner.app (23.1 MB). The first
    attempt stopped at a signing check; the identical second run succeeded.
  - Reference: Mac replay of the street video (finetuned, MPS) gives 16 messages: manhole
    x4, stairs x2, pothole x2, curb, Obstacle x2, fire hydrant x3, car, truck.
- Not tested: install on the iPhone, FPS, speech on silent and the street-video and indoor
  checks on the phone (no iPhone connected; signing needs the user's Apple ID). Android not
  built: no Android SDK on this Mac, and the `.tflite` is only on Drive. Frame-to-audio
  latency on the phone not measured yet.
- Open: cones are still said as "Fire hydrant" when the model is confidently wrong (the naming
  rule only catches doubt or flicker). Korean/Uzbek phrase files not written.

## chore: ignore Core ML packages anywhere in the repo (2026-11-21)

- Why: VS Code showed ~10K uncommitted files. `git status` (including untracked
  files) was already clean: `.venv/` (32,795 files), `.venv-export/` (28,010 files, ignored
  since 6698c6e), `exports/`, `logs/`, `data/`, `.pytest_cache/`, `models/**/*.pt` and
  `__pycache__/` are all ignored. The count was VS Code's stale Source Control view
  from before `.venv-export/` was ignored.
- `.gitignore`: added `*.mlpackage/` and `*.mlmodel`, so a Core ML export outside `exports/`
  (e.g. in `models/`) can't be committed by accident.
- Checked `clear-vision/`: not a git clone, but 2 tracked files from the first commit (the
  original Colab notebook with user-research notes and embedded images, and a copy of
  README.md). Left unchanged until the user decides.
- Tested: `git status --porcelain -uall` empty before and after; `git check-ignore` on a
  sample `models/x.mlpackage/`.

## feat: say "Obstacle" when a post changed class in the last 5 frames (2026-11-21)

- Why: the per-frame naming rule spoke "Traffic cone" / "Fire hydrant" for one object
  whose class flipped between frames at confidence >= 0.6. A name the model keeps
  changing is not trustworthy.
- New rule (third condition of the naming rule): a member of the post group
  (traffic_cone, bollard, fire_hydrant, pole) is spoken as "Obstacle" if the same object
  had more than one class name in the last `name_history_frames` frames, current one
  included. The object is followed with the same matching as the 3-of-5 confirmation
  (new `object_track()`, which `is_confirmed()` now uses too). Only candidates (above
  the speak threshold) count. The warning itself, direction, priority and cooldown key
  are unchanged.
- `configs/default.yaml`: `navigator.name_history_frames: 5` (1 turns the check off).
  `SpeechPlanner` keeps `max(window_frames, name_history_frames) - 1` frames of history;
  confirmation still uses only the last `window_frames - 1`.
- Regenerated `models/nav_config.json` (adds `name_history_frames`) and
  `tests/golden/nav_cases.json`: 35 cases (was 30). The only changed expectation is the
  flicker case, renamed `post_flicker_between_classes_obstacle` ("Traffic cone ahead" ->
  "Obstacle ahead"); checked that every other case's output is identical. New cases:
  hydrant 4 frames back -> Obstacle, 5 frames back -> name kept (exact window boundary),
  two different posts keep their names, a weak hydrant in history is ignored, a pole at
  0.5 then 0.8 keeps its name.
- `docs/navigation_spec.md`: `track()` definition in section 4, third naming condition
  in section 6, history length, config table, known limits.
- Tests: `pytest` 305 passed (was 296): 5 new golden cases, 4 unit tests in
  `tests/test_speech_filter.py` (window boundary, other objects / non-group classes,
  `name_history_frames: 1`, planner history length).
- Latency: `SpeechPlanner.update` measured at 0.68 ms per frame with 20 detections
  (synthetic, 2000 frames); negligible against the 300 ms budget.
- Not tested: a live webcam run with real flickering cones. Open: two touching posts can
  be merged into one track and spoken as "Obstacle" (errs toward warning).

## feat: portable navigation rules (spec, nav_config.json, golden cases) (2026-11-21)

- Why: before building the phone app, the app and Python must use identical rules and
  numbers.
- `docs/navigation_spec.md` describes `src/navigator.py` + `src/planner.py` exactly:
  - inputs and state; geometry formulas (incl. the order of operations for `center_x`)
  - the corridor: the middle third by box center, full height; no other corridor geometry
    exists
  - the per-frame pipeline (candidates, speaking zone, hazards-only, confirmation), the
    risk formula, priorities, cooldowns, message templates, system messages
  - what the live app adds (speech queue), every config key, how to change a rule, known
    limits
- Naming rule for street furniture (budget_v1 calls some cones and bollards "fire
  hydrant", 2026-11-18):
  - `name_groups: {post: [traffic_cone, bollard, fire_hydrant, pole]}`.
  - A group member is spoken as "Obstacle" if its confidence is < `name_min_conf` (0.6),
    or a different group class among the frame's candidates overlaps it with IoU >=
    `name_overlap_iou` (0.5).
  - Direction, distance and priority are never dropped. The group also counts as one
    object for confirmation (otherwise a cone/hydrant flicker would never confirm and the
    warning would be lost) and for the cooldown key (`post:left`, so "Pole, left" then
    "Obstacle, left" is not repeated).
  - The rule is per frame; a class flip across frames keeps the current name.
- `src/navigator.py`:
  - new: `class_group`, `box_iou`, `display_name`, `name_is_doubtful`
  - `message_key` uses the group; `same_object` compares groups
  - `build_phrase` and the templates come from `phrases`; `choose_message(context=...)` and
    `plan_speech` pass the frame's candidates
- `configs/default.yaml`: `name_groups`, `name_min_conf`, `name_overlap_iou`, `phrases`
  (templates and system messages), `spoken_names`. `src/main.py` speaks its system
  messages from `phrases.system` (same texts as before).
- `scripts/export_nav_config.py` → `models/nav_config.json` (committed): the full
  navigator config with defaults filled in, the resolved spoken name of every class, and
  the fine-tuned detection thresholds.
- `scripts/make_nav_golden.py` → `tests/golden/nav_cases.json`: 30 cases (23 speak, 7
  silent), computed with the numbers from `nav_config.json`. They cover:
  - confirmation (steady, flicker, 2 vs 3 of 5, motion, jumping boxes)
  - cooldowns (4 s, 2 s, falling through to the next object)
  - direction incl. the exact left boundary, and proximity/risk
  - speak thresholds, speaking zone, vehicles at 20%, hazards-only, low risk
  - 7 naming cases: confident name, < 0.6, exactly 0.6, two classes overlapping, weak
    overlap ignored, flicker, shared cooldown
- Tests: 296 pass. New `tests/test_nav_golden.py` (34):
  - `nav_config.json` matches the YAML
  - the naming rule is in the JSON
  - 30 unique cases incl. Obstacle, urgent and silent ones
  - Python reproduces every case exactly
  - the stored inputs equal the generator's (no hand edits)
  - Drift check: `name_min_conf` 0.61 makes exactly `post_name_at_exactly_min_conf` fail.
  - `test_speech_filter`: pole/bollard is now one object, so that test uses car/truck; new
    test that cone and hydrant confirm each other.
- Not tested or open:
  - The phone app's implementation of the spec (its golden test).
  - The live app with the naming rule on a real street.

---

## fix: camera reconnect loop and short webcam hiccups (2026-11-20)

- Why: in the first live webcam runs on macOS 27 (MPS), the app said "Camera lost" /
  "Camera back" about every 4 s and processed only ~3-14 frames/s.
- Findings (webcam runs of 30 s each, fine-tuned budget_v1, logs in `logs/`):
  - CPU instead of MPS: the same dropouts ("Camera lost" spoken 7x in 30 s, ~2
    frames/s). The GPU and the camera do not interfere.
  - OpenCV 5.0.0.93 instead of 5.1.0.95 (upgraded in `.venv` on 2026-11-19, not by these
    changes), same old code on MPS: 5x in 30 s, 3.7 frames/s. Not the cause.
  - Our bug: on a dead camera, `CameraSupervisor` opened a second capture on the same
    webcam while the old one was still open, and released the old one after the swap.
    macOS then killed the new capture about 1 s later, so one short hiccup became an
    endless lost/back loop.
  - The remaining stalls came from index 0 being the iPhone as a Continuity Camera
    (`system_profiler`: FaceTime HD Camera + iPhone_zk Camera; index 0 defaults to
    1920x1080, which the M1's built-in camera can't do).
    - In a bare OpenCV script, index 0 delivered ~150 frames, then nothing for 13 s.
    - Index 1 (built-in camera) gave a steady 30.1 frames/s for 20 s, longest gap 0.12 s.
- Fix:
  - `src/main.py` `CameraSupervisor._reconnect` releases the old camera before opening the
    new one, so there is never more than one capture per webcam.
  - `src/camera.py`: failed reads are retried quietly (debug log). `healthy` (and so
    "Camera lost") turns False only after `stale_timeout` without a frame, now 1.5 s
    (`DEFAULT_STALE_TIMEOUT_S` and `camera.stale_timeout`). A recovery is logged as a
    warning only if the gap reached that threshold. "Camera back" is still spoken once on
    recovery.
  - `src/main.py`: SIGTERM stops the app like Ctrl+C. Background and scripted runs ignore
    SIGINT, so the first 30 s CPU test ran 147 s.
  - `configs/default.yaml`: a note on the camera index (an iPhone can be index 0).
    `source` stays 0, because the index order changes when the iPhone is away.
  - `requirements.txt` and `.venv`: `opencv-python==5.0.0.93`.
- Results, fixed code, MPS, 30 s:
  - iPhone camera (index 0): "Camera lost" 3x (was 5-7), 14.0 frames/s (was 3.7). The
    reconnect loop is gone; the left-over stalls are the iPhone's.
  - Built-in camera (`--source 1`), OpenCV 5.0.0.93: 0 dropouts, 19.2 frames/s over the
    first 30 s incl. start (~27/s in steady 5 s windows), model 18-23 ms, frame-to-decision
    36-50 ms on average.
  - Built-in camera, OpenCV 5.1.0.95: 0 dropouts, 14.8 frames/s (~23.6/s steady), model
    20-31 ms. So 5.0.0.93 stays.
  - No MPS errors or warnings in any run.
- Tests: 261 pass. New `tests/test_camera.py` (6), with a scripted fake VideoCapture:
  - a 0.3 s hiccup stays healthy; a 1 s gap is lost about 0.5 s after the last frame and
    comes back
  - the 1.5 s default and config
  - the supervisor never has two cameras open across 5 reconnects (fails with the old code)
  - "Camera lost"/"Camera back" are spoken exactly once each
  - SIGTERM behaves like Ctrl+C
- Not tested or open:
  - Picking the built-in camera by name instead of index (OpenCV has no device names;
    needs AVFoundation).
  - Longer webcam runs while walking.

---

## feat: Core ML export on the Mac with its own environment (2026-11-20)

- Why: in Colab (Python 3.13), notebook 07's Core ML export failed with `TypeError: only
  0-dimensional arrays can be converted to Python scalars` in coremltools'
  `frontend/torch/ops.py` `_cast`.
  - Cause: numpy 2.4+ breaks coremltools 9.0's PyTorch converter (coremltools issue #2633,
    fix merged Jan 2026 but released only as a 9.1 pre-release).
  - Each export runs in its own process, but numpy 2.4 was already imported before
    Ultralytics' numpy downgrade, so the export still ran on numpy 2.4.
- Export environment `.venv-export` (gitignored, separate from the app's `.venv`):
  - Homebrew Python 3.11.15, built from wheels only, recorded in `requirements-export.txt`.
  - ultralytics 8.4.207 (same as the app), coremltools 9.0 (latest stable; arm64 wheels
    for Python 3.8-3.13), torch 2.7.0 + torchvision 0.22.0 (coremltools 9.0: "Torch 2.7.0 is
    the most recent version that has been tested"), numpy 2.3.5 (< 2.4).
- `scripts/export_coreml.py --run budget_v1 --images data/street_photos`:
  - Exports `models/<run>/best.pt` at the imgsz from `metrics.json` (512), FP16
    (`quantize=16`), `nms=True`, with NMS confidence 0.15 (the lowest per-class threshold),
    from a temporary copy of the weights.
  - Plugin settings: the `ultralytics_yolo` docs say "nms=True embeds NMS where supported".
    The official iOS assets use `nms=False` only because they are NMS-free YOLO26 models;
    our YOLO11n needs NMS.
  - On the numpy TypeError it retries once with `nms=False`; not needed in the final run.
  - Writes `exports/<run>/mobile/clear_vision_<run>.mlpackage`, `.mlpackage.zip` and
    `coreml_metrics.json` (`exports/` is gitignored). A missing `best.pt` gives the Drive
    path to download from, and numpy >= 2.4 is refused with the issue number.
  - Checks the embedded metadata (task detect, the 20 classes, imgsz) and compares Core ML
    with `best.pt` image by image: same class and IoU > 0.5, one-to-one, share = 2·pairs /
    all boxes, pass >= 95%, median time per image.
  - With no images it reports NOT CHECKED, not PASS; the first draft printed PASS.
- Result for budget_v1 (macOS 27.2.1, M1):
  - Exported first try with `nms=True` (4.5 s). `.mlpackage` 5.4 MB, zip 4.9 MB, metadata OK.
  - 24 Wikimedia street photos (re-downloaded to `data/street_photos/`, gitignored), boxes
    at conf >= 0.25: 47 of 49/48 boxes paired, 96.9%, PASS. The only differences are 1-2 small
    cars near 0.25 in two crowded scenes.
  - Speed: Core ML 12.1 ms per image, best.pt 41.6 ms.
  - That comparison feeds best.pt the same 512x512 square input as Core ML (`rect=False`).
    With Ultralytics' default rectangular letterbox for the .pt (`--rect-pt`) the share was
    92.8%; the input shape alone moves borderline scores. On the iPhone the plugin feeds the
    fixed square input, so the square comparison is the relevant one.
  - No test-split images exist locally (merged.zip is on Drive), so the 50-test-image
    check was not run; `--test-images` supports it.
- PyTorch MPS on macOS 27.2.1 now works (tensor ops, conv and YOLO predict, each in its own
  process; same detections as the CPU):
  - `.venv` (torch 2.15): 20.1 ms on MPS vs 25.5 ms on the CPU per 512 image. The app's
    `device: auto` now picks MPS.
  - `.venv-export` (torch 2.7): 30.0 ms on MPS vs 26.5 ms on the CPU.
- Notebook 07: Setup installs `"numpy<2.4" "coremltools>=9.0"` with ultralytics before any
  export process starts. The header points to the Mac route.
- CLAUDE.md: commands for `.venv-export`. `.gitignore`: `.venv-export/`, `exports/`, `data/`.
- Tests: 255 pass. New `tests/test_export_coreml.py` (5): one-to-one matching (class, IoU,
  no double pairs), the match share incl. budget_v1's 96.9%, the missing-weights message, the
  numpy 2.4 guard, the image listing.
- Not tested or open:
  - The notebook 07 numpy pin in Colab.
  - The live app on MPS (device auto) with latency.
  - Loading the `.mlpackage.zip` in the Flutter plugin on an iPhone.
  - Uploading the three files to Drive `exports/budget_v1/mobile/` (manual).

---

## feat: mobile export notebook (TFLite, Core ML, Flutter plugin files) (2026-11-18)

- Why: run the fine-tuned detector on phones, and give the Flutter app
  (`ultralytics_yolo`) the files it expects.
- `notebooks/07_export_mobile.ipynb` (T4):
  - `RUN_NAME` is the only setting a later model needs. imgsz comes from the run's
    `metrics.json` (budget_v1: 512).
  - The data comes from `prepare_dataset()`; images are never read from Drive.
  - Setup has all imports, the `is_kaggle()` rule and the "Google Drive is not connected"
    check. The model's class names must match `merge_report.json`.
  - Exports, each as its own process:
    - TFLite INT8: `format="litert"`, `quantize=8`, calibrated on 200 random train images.
    - TFLite FP16: `format="saved_model"` → `*_float16.tflite`.
    - Core ML FP16: `nms=True`, `quantize=16`, `conf=0.15`.
    - Flutter Android: `format="litert"`, `nms=None`, FP32 → `clear_vision_<RUN>.tflite`.
    - Flutter iOS: `clear_vision_<RUN>.mlpackage` plus `.mlpackage.zip`.
  - Checks the embedded metadata (task, 20 class names in order, imgsz) that the plugin
    reads.
  - Validates every model on the test split with notebook 08's evaluator: mAP50, recall of
    curb/stairs/pothole/pole/manhole at their own thresholds, size vs the `.pt`, and
    PASS/FAIL for FP16 at < 3% mAP50 loss.
  - Saves everything to `exports/<RUN>/mobile/` with `labels.txt`, `thresholds.json` (0.25
    default, 0.15 critical) and `mobile_metrics.json`.
- Findings that shaped it (Ultralytics 8.4.207):
  - There is no `tflite` format; TFLite is `format="litert"` and precision is `quantize`.
    LiteRT has no FP16 (only 8, w8a16, w8a32), so the FP16 file comes from the TensorFlow
    route.
  - That FP16 file is NHWC with pixel box coordinates, but Ultralytics' own LiteRT reader
    assumes normalized coordinates. `val()` scored it 0.000 while its raw scores matched
    FP32 (0.545 vs 0.547). `scripts/mobile_export.py` `TFLiteDetector` reads every
    `.tflite`, checking NHWC/NCHW, pixel/normalized and float/int8 input instead of
    assuming them.
  - Core ML's embedded NMS defaults to confidence 0.25, which would drop curb, stairs,
    pothole, pole and manhole boxes between 0.15 and 0.25 on iPhone. It is exported with
    `conf` = the lowest per-class threshold (0.15), still adjustable at run time.
  - Exports install packages mid-run; in one process torch went 2.15 → 2.14.1 and numpy
    2.4.6 → 2.1.3 while loaded, and all four exports failed. Each export and each
    validation now runs as its own process (`python scripts/mobile_export.py
    export|evaluate`).
  - Flutter plugin docs (`yolo-flutter-app` `doc/models.md`, package 0.7.2): Android
    `litert` with `nms=None` (NMS in the plugin), FP32 or W8A32; iOS `.mlpackage`, or
    `.mlpackage.zip` in `assets/models/`. Task and labels come from the embedded metadata.
    The custom Core ML export arguments are not documented there.
- Rehearsed on this Mac:
  - The notebook's real cells, sections 1-6, ran on the real `budget_v1/best.pt`, in a
    separate throwaway venv (Ultralytics 8.4.207 plus the export packages it installs).
  - The test set was a mini `merged.zip` of the 24 Wikimedia street photos, labelled with
    the `.pt`'s own detections at conf >= 0.5 (30 boxes).
  - mAP50 vs the `.pt` (1.000): INT8 0.986 (-1.4%), FP16 1.000 (0%), Core ML FP16 0.998
    (-0.23%), Flutter LiteRT FP32 1.000. Recall 1.0 for every critical class present.
  - Sizes: INT8 3.1 MB, FP16 5.3 MB, Core ML 5.4 MB (zip 4.9 MB), Flutter FP32 10.7 MB,
    `.pt` 5.5 MB.
  - Metadata OK in all four.
- Tests: 250 pass. New `tests/test_mobile_export.py` (13):
  - imgsz from `metrics.json` (budget_v1 = 512), the calibration yaml (200 of 250, val =
    train list), thresholds from the config, side files, the zipped `.mlpackage` layout
  - TFLite metadata in the appended zip, both name flavours, result rows and sizes,
    replacing copies
  - command-line option parsing, Core ML skipped outside macOS
  - the decoder with pixel/normalized and either orientation plus NMS, letterbox, recall at
    per-class thresholds
- Not tested or open:
  - The real Colab run on the full test split (the acceptance numbers).
  - Core ML is scored only on macOS; in Colab it is exported and its metadata checked.
  - Loading the files in the Flutter plugin on a phone.
  - The plugin applies one global threshold, so the app has to apply `thresholds.json`
    itself.

---

## fix: notebook 05 Kaggle detection and standalone evaluation sections (2026-11-18)

- Bug 1:
  - Notebook 05 detected Kaggle with `Path("/kaggle").exists()`. Colab runtimes now have a
    `/kaggle` folder too, so in Colab the notebook cloned the repo to
    `/kaggle/working/clear-vision` and skipped `drive.mount`.
  - `prepare_dataset` then reported `merged.done is missing` although the file is on Drive.
- Fix:
  - `src/colab_data.py` `is_kaggle()`: true only if `KAGGLE_KERNEL_RUN_TYPE` is set and
    neither `google.colab` is loaded nor `COLAB_RELEASE_TAG` is set.
  - The Setup cell has to choose the clone folder before the repo can be imported, so it
    keeps a one-line copy of the same rule. After the import it asserts
    `ON_KAGGLE == is_kaggle()`. No other notebook or script tested `/kaggle`; checked with
    grep over src, scripts, notebooks and tests.
- Setup fails early with clear messages:
  - In Colab: "Google Drive is not connected: <DRIVE_ROOT> does not exist" when
    `DRIVE_ROOT` is missing after `drive.mount`.
  - On Kaggle: when the dataset input is missing. `DRIVE_ROOT` under `/kaggle/working` is
    created.
- Bug 2: with training already finished and sections 3/3b skipped, section 4 failed with
  `NameError: YOLO`.
  - All imports of sections 3-7 (YOLO, pandas, yaml, cv2, matplotlib, IPython Image, csv,
    `compare_detectors`, `Detector`) are now in the Setup cell.
  - Section 4 stops with a clear message if `best.pt` doesn't exist.
  - A note above section 4: "Training already done? Skip sections 3 and 3b and run from
    here." The header lists the same route.
- Tests: 237 pass. `tests/test_colab_data.py` has 8 new `is_kaggle` cases: Kaggle
  interactive/batch, local, an empty variable, Colab with Kaggle variables (by tag and by
  module), and an existing `/kaggle` folder.
- Ran the notebook's real cells locally (Colab parts stubbed: `google.colab.drive.mount`,
  clone, `%pip`, GPU check) against the CPU smoke run from 2026-11-16.
  - Settings → Setup → data → sections 4, 4b, 5, 6, 7 ran without the training or resume
    cells and wrote `best.pt`, `best.onnx`, `metrics.json` and `per_class_choice.yaml`.
  - A missing `DRIVE_ROOT` gives the new message; `KAGGLE_KERNEL_RUN_TYPE` takes the Kaggle
    branch.
- Not tested: in a real Colab or Kaggle runtime.

---

## feat: speech noise filter and replay script (2026-11-18)

- Why: in the first live test indoors, the app said "car ahead" and "pole, right, close"
  with no cars or poles around. The model knows only outdoor street scenes, so it forces
  walls, door frames, columns and furniture into our classes, and every single-frame guess
  was spoken at once. This is a light filter before the full tracker (Phase 4).
- `src/navigator.py` (pure functions, new `NavigatorConfig` fields):
  - `speak_threshold` / `confident_enough`: per-class speak thresholds (vehicles 0.45,
    person 0.35, pole/bollard/traffic_cone/fire_hydrant 0.25, curb/stairs/pothole/manhole
    0.20, others 0.35). These are separate from the detection thresholds, so the debug
    window still shows weak boxes.
  - `same_object` / `is_confirmed`: speak only if the same class was seen in >= 3 of the
    last 5 frames in roughly the same place (centers <= 0.15 apart, IoU >= 0.3 with the
    previous match). Gaps don't break the chain, and a slowly moving object stays the same
    object.
  - `in_speaking_zone`: box bottom in the lower 60% of the frame, or box height > 15%.
    Vehicles only when taller than 20%.
  - `allowed_class`: with `mute_non_hazards`, only `hazard_classes` are spoken.
  - `plan_speech()` returns candidates, speakable boxes and the message.
  - Cooldown: 4 s per class + direction (unchanged); urgent repeats after 2 s (was 1.5).
- `src/planner.py` `SpeechPlanner`: the frame history, last-spoken times and the
  hazards-only toggle, shared by `main.py` and `scripts/replay.py`.
- `src/main.py`:
  - Uses the planner.
  - Debug window: weak boxes gray, confident-but-unconfirmed boxes thin, speakable boxes
    thick, and a horizon line.
  - Key `m` toggles hazards only and says "Hazards only" / "All objects".
- `scripts/replay.py`:
  - Plays a video file on its own clock (frame / fps), or `--source 0` for the webcam, and
    prints every message with its time and a count per class.
  - `--no-filter` gives the old behaviour for before/after comparisons. Also `--step`,
    `--seconds`, `--show` and `--hazards-only`.
- Results (fine-tuned model, replay of every frame):
  - 24-photo street video (48 s): 28 messages without the filter, 16 with it. Stairs (both
    staircases), potholes (2), manholes (4) and curbs (1) are still spoken; the dropped
    stairs message was only the 1.5 s urgent repeat. Cones still come out as "Fire
    hydrant".
  - Indoor stand-in: a 60 s clip of 9 panned Wikimedia room photos (classrooms, corridor,
    hotel lobbies, living rooms, offices). 13 messages before, 5 after:
    - 1 correct: a real person in an office.
    - 1 false: a Buddha painting as person (0.37).
    - 3 false: a painted column as barrier (0.43), spoken three times because its
      direction changed from right to ahead while panning, plus the 2 s urgent repeat.
  - Live with the debug window: 20.7-21.3 FPS, frame-to-decision 47-48 ms (unchanged).
    Confirmation delays the first warning of a new object by ~2 frames (~100 ms, ~145 ms
    in total).
- Tests: 229 pass. New `tests/test_speech_filter.py` (26):
  - config values
  - flicker (1 of 5) silent; 2 of 5 silent, 3 of 5 spoken; steady object spoken once;
    slow motion confirmed; jumping boxes not; another class in the same place doesn't
    count; only the last window counts
  - 4 s normal and 2 s urgent repeats
  - speak thresholds on both sides for car, person, pole, curb and bench; weak boxes not
    candidates
  - horizon and size zone, vehicles at 20%, hazards-only toggle
  - `replay.py` on a generated clip (video times, flicker silent, `--no-filter`, `--step`,
    `--source`)
- Not tested or open:
  - The real webcam room test (acceptance: at most 1-2 messages in 60 s). The stand-in
    clip still gives 4 false ones.
  - Indoor columns as "barrier" are the main remaining source. Raising the barrier speak
    threshold would also silence real street barriers (0.29-0.40 on the photos).
  - A tracker (Phase 4) would stop direction changes from creating new cooldown keys for
    the same object.

---

## feat: fine-tuned detector in the live app (2026-11-18)

- Why: budget_v1 (notebook 05) beat the zero-training hybrid on 18 of 20 classes on the
  test split. Only dog (0.830 vs 0.832) and bench (0.391 vs 0.426) were slightly better
  with yolo11n.
- `src/detector.py`:
  - New `finetuned` backend: `models/budget_v1/best.pt` via Ultralytics at imgsz 512
    (configurable). Class names come from `configs/classes.yaml` by ID; a different class
    count is an error, different names a warning. ONNX stays optional (`use_onnx`).
  - `hybrid` (now the default backend):
    - With `best.pt` and `finetuned.only: true` (FINETUNED_ONLY, the default): the
      fine-tuned model alone. yolo11n and YOLOE are never loaded.
    - With `only: false`: `resolve_choice()` reads `per_class_choice.yaml`. A
      zero-training model (`zero_training_model`: coco = yolo11n, openvocab = YOLOE) gets a
      class only if it won by at least `min_gain` (MIN_GAIN, 0.05). yolo11n and YOLOE load
      only if a class still needs them, and results are merged with NMS as before.
    - Without `best.pt`: the zero-training hybrid, with a log warning.
  - `HybridDetector` takes several every-frame models, and the open-vocab model is
    optional.
  - Fine-tuned thresholds: 0.15 for curb, stairs, pothole, pole and manhole (a miss is
    worse than a false alarm; at 0.25 the test split missed 42%, 34%, 36%, 37% and 23% of
    them), 0.25 for the rest.
- `configs/default.yaml`: `backend: hybrid` and a new `finetuned:` section.
- `.gitignore`: `models/**/*.pt` and `models/**/*.onnx`. `per_class_choice.yaml` and
  `metrics.json` of budget_v1 are committed.
- MPS: still crashes on this Mac (macOS 14.5, an uncaught NSException aborts Python; tried
  in a separate process). `device: auto` keeps using the CPU and switches to MPS by itself
  on macOS 15+. The CPU is fast enough: best.pt takes 30 ms at 512 (44 ms at 640).
- With MIN_GAIN 0.05 no class leaves the fine-tuned model: bench wins by 0.035 and dog by
  0.002, both under 0.05. Bench goes to yolo11n only at `min_gain` <= 0.035.
- Live, on the same 720p hazard video as on 2026-11-16, with the debug window, default
  backend:
  - 20.9-21.8 FPS, inference 30-32 ms, frame-to-decision 46-48 ms on average (max 91 ms).
  - The zero-training hybrid did 11.0-11.6 FPS and 86-91 ms.
- Real-world check on the 24 Wikimedia photos (out of distribution):
  - Better than zero-training on manholes (4/4 vs 1), stairs (3/3 vs 2) and potholes (3/4
    vs 1).
  - Worse on traffic cones (0/4 vs 2; two are called fire_hydrant at 0.81 and 0.71, as
    yolo11n's COCO head did), bollards (1/4 vs 3; one fire_hydrant at 0.91) and poles
    (1/4 vs 3).
  - The 0.98 cone mAP50 on the test split doesn't carry over to these photos. The test
    split comes from the same Roboflow sources as the training data. The objects are still
    announced, but under the wrong name.
- Tests: 203 pass. New `tests/test_finetuned.py` (14, fake models):
  - `resolve_choice` with budget_v1's real scores at min_gain 0.05 / 0.03 / 0 and YOLOE
    routing
  - fine-tuned names, imgsz and thresholds; a wrong class count; a missing file
  - hybrid: only=true, only=false with 0.05 (still one model), lost classes to yolo11n
    only, YOLOE loaded only for a class it won, fallback without best.pt and its log
  - several fast models
  - `test_openvocab.py`'s zero-training hybrid test now sets `finetuned.model_path: null`.
- Not tested or open:
  - The live webcam.
  - The real per-class hybrid with yolo11n/YOLOE (only with fakes; with today's scores it
    never triggers).
  - The cone/bollard/pole regression on real street photos needs more varied training data
    or a per-class override.
  - The rest of Part A's spec was cut off in the request after the thresholds bullet.
  - chair, potted_plant and stop_sign, which only yolo11n knew, are no longer detected with
    the fine-tuned model.

---

## feat: light fine-tune notebook for the free T4 (2026-11-16)

- Why: notebook 08's zero-training hybrid misses key hazards. Its test mAP50: curb 0.001,
  barrier 0.01, pothole 0.03, crosswalk 0.04, pole 0.08, manhole 0.08, bollard 0.18,
  scooter 0.24, stairs 0.31; only traffic_cone (0.77) was good. This fine-tunes the
  COCO-pretrained yolo11n (not from scratch), within the free Colab T4 budget.
- `notebooks/05_train_yolo.ipynb` (T4):
  - Presets:
    - BUDGET (default): yolo11n, 512 px, 30 epochs, 30% subset, freeze=10, batch=-1,
      patience 10.
    - NORMAL: 640 px, 50 epochs, 50% subset, no freeze, patience 15.
  - Data: `prepare_dataset()` then `apply_balanced()`. If the balanced list is stale or
    missing, it stops with a message: re-run notebook 03 or set `USE_BALANCED = False`.
    Training never reads from Drive.
  - Subset (`train_subset.txt` + `data_subset.yaml` in the local dataset): every image
    with a focus class is kept, with its balanced repeats. `SUBSET_FRACTION` of the rest is
    kept, and person-only images are capped at `PERSON_CAP` (2000). It is seeded, so a
    resume rebuilds the same list. Val and test are untouched.
  - Train: `project=MyDrive/clear_vision/runs`, `save_period=5`.
    - Augmentations: hsv, ±5° rotation, scale 0.5, mosaic with `close_mosaic=5`, flipud off.
    - Checks for 0.5 GB of free Drive space first.
    - Refuses to start over a run that already has weights. `exist_ok=True` stops
      Ultralytics from renaming the run.
    - `EpochTimer` prints the time of the first epoch and an estimate for the rest.
  - Resume cell: `YOLO(last.pt).train(resume=True)`.
  - Test split:
    - Ultralytics `val(split="test")` gives recall and the confusion matrix.
    - For the comparison table, the fine-tuned model is scored with notebook 08's own
      evaluator (`compare_detectors.ap50_per_class`, conf 0.01). Both columns are then the
      same metric.
    - Table: class | zero-training hybrid | fine-tuned | better? | recall. Recall is
      highlighted for curb, stairs, pothole, manhole and pole.
    - `RECOMPUTE_ZERO_TRAINING = True` re-scores the hybrid on today's labels (the
      notebook 08 JSON may predate the pseudo-labels).
  - `per_class_choice.yaml`: `choice: {class: finetuned | openvocab}`, plus the scores and
    `zero_training_model` (`coco` for the classes the hybrid takes from yolo11n).
  - Missed critical objects: false negatives at conf 0.25 (count per class, grid of 12:
    missed box red, predictions gray, "seen as X" when another class was predicted there).
  - Export: `best.pt`, optional `best.onnx` and `metrics.json` to `exports/<RUN_NAME>/`.
  - A Kaggle section. The setup cell detects `/kaggle` and skips the Drive mount.
- `scripts/train_utils.py`: presets, augmentations, `build_subset`, `write_subset`,
  `EpochTimer`, `predict_split`, `find_misses`, `miss_grid`, `comparison_rows`,
  `per_class_choice`, `check_space`, `per_class_from_ultralytics`.
- Tests: 189 pass. New `tests/test_train_utils.py` (16):
  - presets
  - subset: focus kept with repeats, fraction, person cap, determinism, and real
    `check_det_dataset` on the written yaml
  - timer (once, after resume), space check
  - misses (low confidence, wrong class), grid
  - comparison and choice (yes/no/same/n-a, COCO source), the notebook 08 JSON, the
    Ultralytics metrics adapter
- Smoke test on the Mac CPU with the fixture dataset and the notebook's exact train
  arguments (6 epochs, 64 px, freeze=10, save_period=1):
  - The training process was killed with `kill -9` after epoch 3. The resume cell's code
    continued "from epoch 4 to 6" and finished.
  - Then ran the notebook's own cells for sections 2 and 4-7 against that run: the
    balanced-list refusal message, the "already has weights" guard, test val, the
    confusion matrix file, the comparison table, `per_class_choice.yaml`, the misses, and
    `best.pt`, `best.onnx` and `metrics.json` in exports.
  - Scores were 0, as expected from 3 tiny images.
- Not tested or open:
  - The real T4 run, so the 1-2 h target is an estimate. The subset keeps every focus image
    plus its balanced repeats, and most of the dataset comes from focus-class sources, so
    the subset may be much more than 30% of train. If the estimate printed after epoch 1 is
    too long, lower `EPOCHS` or `SUBSET_FRACTION`, or set `USE_BALANCED = False`.
  - Run size on Drive: checkpoints were ~16 MB each in the smoke test, so 30 epochs gives
    about 130 MB.
  - Kaggle steps were written from knowledge of the platform, not run.
  - The app doesn't read `per_class_choice.yaml` or `best.pt` yet.

---

## feat: dataset report notebook with sanity checks and balanced train list (2026-11-16)

- Why: see what's in the merged dataset, catch bad mappings and split leaks, and give
  training a balanced train list before the first real training run.
- `scripts/dataset_report.py`:
  - `scan_dataset()` reads every label file and image header from the local unpack.
    Each box gets its split, source (file name prefix), class, size and a pseudo flag
    (the last N lines of a label file, where N is the image's row count in
    `pseudo_labels.csv`; if the CSV doesn't match the file, it's counted as original with
    a warning).
  - Box sizes use COCO bins (small < 32², medium < 96²), measured at the 640 px training
    size.
  - `sanity_checks()`:
    - FAIL on an image in two splits (by name and by md5) or class IDs outside 0..19 or
      malformed lines.
    - FAIL on an image count that differs from `merge_report.json`.
    - WARN on empty label files in val/test. The merge keeps background images on purpose,
      so they aren't broken.
  - `balanced_train()`:
    - An image with a focus class is listed `ceil(target / train images with the class)`
      times, at most 4. The target is the median train images per class.
    - Person-only images are randomly capped at 4,000 (seeded). Everything else is listed
      once.
    - Lines are `./images/train/<name>`; Ultralytics resolves `./` against the list's
      folder.
  - `write_balanced()` writes `train_balanced.txt` and `data_balanced.yaml`. The yaml
    stores merged.done's content and the balance settings.
  - `example_grid()`: 4x4 images of one class. Its original boxes are yellow, its
    pseudo-labels orange, other classes thin gray.
- `src/colab_data.py` `apply_balanced()`: after `prepare_dataset()`, it copies the list into
  the dataset and sets `path`. It refuses a list made for another merged.zip
  (merged.done differs) or one that names a missing image.
- `notebooks/03_dataset_report.ipynb` (CPU):
  - Unpacks with `prepare_dataset()` and never reads images from Drive.
  - Charts: images per split; boxes per class per split (original vs pseudo, stacked);
    size share per class; boxes per source per class (top 7 + Other, with a table).
  - An ipywidgets class dropdown with shuffle.
  - The class table ("needs more data" under 300 train boxes) and the sanity checks.
  - Writes the two balanced files next to merged.zip, but refuses if a check FAILed.
  - Chart colours come from the dataviz skill's default palette. The validator passes;
    three slots are under 3:1 contrast, so the source chart gets a table.
- Tests: 173 pass. New `tests/test_dataset_report.py` (16):
  - size bins, source prefix, scan counts against the report
  - pseudo counted separately (after a fake notebook-09 run)
  - the class table
  - every sanity check, failing on an injected fault: a copy in another split, the same
    name in two splits, ID 19 and a malformed line, an empty val label
  - repeat factors and the cap, person-only cap, determinism
  - `apply_balanced()` + `check_det_dataset` + a real Ultralytics `YOLODataset` that keeps
    the repeated lines
  - stale or missing lists refused, the grid
- Also ran every notebook code cell except Colab setup locally on the pseudo-labeled
  fixture zip, and looked at the charts.
- Not tested or open:
  - The real run in Colab. md5 of ~5.7 GB is likely a few minutes; set
    `HASH_IMAGES = False` to skip it.
  - The ipywidgets dropdown only ran headless, not clicked.
  - The training notebook doesn't exist yet. It should call `apply_balanced()`.
  - Ultralytics writes `labels/train.cache` for both the balanced and the plain list, so
    switching between them rescans once.
  - Repeating focus images also repeats the other boxes in them.

---

## feat: pseudo-label missing COCO objects in merged.zip (2026-11-16)

- Why: each source labels only its own classes (people in rf_cone images aren't boxed).
  Training learns to ignore those objects. In notebook 08, correct person and car
  detections in non-COCO images also counted as false positives (person 0.39, car 0.21
  mAP50).
- `scripts/pseudo_label.py`:
  - Loads the data with `prepare_dataset(merged.zip)` into `/content` (never reads images
    from Drive) and loads yolo11x once.
  - Predicts person, bicycle, car, motorcycle, bus, truck, dog, bench, fire hydrant and
    traffic light, mapped to our class IDs, on every image of a non-COCO source. All
    splits are covered. The source comes from the file name prefix, with the longest source
    name winning. `coco_subset` and `openimages` are skipped (configurable).
  - A box is added only if its confidence is >= 0.5 and no existing box of the same class
    overlaps it with IoU > 0.5. Both values are configurable.
  - Original lines are never touched; new lines are appended.
  - Writes `labels_original.zip` (original content of only the changed label files) and
    `pseudo_labels.csv` (image, split, source, class, confidence, box). Updates
    `merge_report.json` with `pseudo_labels` (settings, totals, per source, per split) and
    `pseudo_labels_added` per source. `boxes` still counts only original labels.
  - Upload order: the backup and CSV are uploaded first, flushed and verified, so merged.zip
    is never changed without a working backup. Then `publish_zip()` does the same as append
    mode: space check +10%, delete merged.done, upload, flush, verify size and CRC, write
    merged.done.
  - Re-running skips if the report has `pseudo_labels`. `force` restores the originals from
    `labels_original.zip`, then labels again. The new backup is the union of old and new,
    so it can always undo every earlier run.
  - The local `.prepared` marker is deleted before labels change, and rewritten with the
    new merged.done after a verified upload. A crash mid-run makes `prepare_dataset()`
    unpack fresh instead of reusing half-edited labels.
  - Grid of 16 random changed images (original green, added yellow). `changes_from_csv()`
    rebuilds it on a skipped run.
- `scripts/merge_datasets.py`: `make_zip` no longer packs `.prepared` or Ultralytics
  `*.cache` files. This matters for any zip built from a `prepare_dataset()` unpack.
- `notebooks/09_pseudo_labels.ipynb`: T4 GPU. Settings (`CONF`, `IOU`, `COCO_SOURCES`,
  `FORCE_PSEUDO`), then unpack, pseudo-label, the added-boxes table (also per split), the
  grid, and a final `prepare_dataset()` that runs `check_det_dataset`.
- Tests: 157 pass. New `tests/test_pseudo_label.py` (17), on a real merged.zip built from
  the merge fixtures, with a fake model:
  - overlap skip vs. another class, the 0.49/0.50 threshold, background and test-split
    images
  - originals kept byte for byte, backup = changed files only, CSV and report (Drive copy
    and zip copy)
  - merged.done deleted before the zip upload and written after; no `.prepared` in the zip
  - `prepare_dataset` + `check_det_dataset` after the run
  - COCO sources skipped
  - re-run skip, force restore (no doubled boxes, backup still covers run 1), force without
    a backup fails
  - a lost backup upload leaves merged.zip and merged.done alone
  - crash → fresh unpack
  - grid, and the grid rebuilt from the CSV
- Also ran notebook sections 2-6 locally on the fixture zip with the real yolo11n, with a
  street photo in a src_b image. 7 people and cars were added after the original cone line.
  The re-run skipped, and force gave the same 7 boxes. The grid was checked by eye.
- Not tested or open:
  - The real run in Colab on the 5.7 GB zip (time, Drive space, the added counts). yolo11x
    on a T4 is roughly 25 ms per image at 640.
  - Pseudo-labels are model guesses. False positives at 0.5 (the grid helps review them)
    also enter val/test, so mAP measures agreement with yolo11x on those boxes.
  - Notebook 08 should be re-run afterwards to compare mAP.
  - An append (notebook 02) after this leaves new images un-labeled, and a re-mapped source
    gets its raw labels back. The report still says "done", so re-run with
    `FORCE_PSEUDO = True`.

## feat: zero-training open-vocab and hybrid detector backends (2026-11-16)

- Why: detect sidewalk hazards (stairs, cones, poles, potholes, ...) before any
  fine-tuning, using a text-prompted open-vocabulary model.
- `src/detector.py`:
  - Backends: `ultralytics | onnx | openvocab | hybrid`, chosen by `detector.backend` or
    `python -m src.main --backend ...`. `build_detector()` builds any of them.
  - `OpenVocabDetector`: YOLOE `yoloe-11s-seg.pt`, falling back to YOLO-World
    `yolov8s-worldv2.pt` if YOLOE is missing from Ultralytics or fails to load.
    - `set_classes` runs once with the phrases from the new `configs/prompts.yaml`. A class
      may have several phrases, and they all map back to our class name. Masks are
      ignored; it returns plain `Detection`s.
    - The prompted model is saved to `models/openvocab/<model>-<hash>.pt`. The hash covers
      the model, phrases and mapping. Later starts load it in about 1 s with no text
      encoder (the first prompting needs the ~600 MB MobileCLIP download).
    - `runtime: onnx` exports the prompted model to ONNX after `set_classes`, so it runs
      without the text encoder.
  - `HybridDetector`: yolo11n every frame for the COCO classes, plus YOLOE every
    `openvocab_every` frames for the 10 non-COCO classes. Between runs it reuses the last
    YOLOE boxes ("when unsure, warn"). The two are merged with per-class NMS, then an
    agnostic NMS (IoU 0.7) that ranks by confidence divided by the class threshold.
    yolo11n calls bollards and cones "fire hydrant", so the open-vocab label usually wins.
  - Per-class thresholds `class_conf` (COCO 0.35, rare 0.15, pole 0.12), plus `conf` 0.4
    for classes not listed.
  - All backends output `configs/classes.yaml` names (`fire_hydrant`, not
    `fire hydrant`). `select_device("auto")` now also picks CUDA (Colab).
- `configs/default.yaml`:
  - The `detector:` section is restructured into `coco:`, `openvocab:` and `hybrid:`.
  - The default backend stays `ultralytics`.
  - Severity entries for the new classes: stairs, pothole and scooter 3; pole, bollard,
    cone, barrier, curb and manhole 2. Keys renamed to underscores.
- `src/navigator.py`: speaks underscores as spaces ("Traffic cone, left").
- `src/main.py`:
  - `--backend` flag.
  - If the openvocab or hybrid backend fails to load, it falls back to COCO and says
    "Hazard detector failed. Detecting people and vehicles only."
  - FPS bug fixed: it averaged 1/interval, which read ~14 FPS for hybrid's alternating
    frames when the real rate was ~11. It now averages the interval and logs the frame
    count per stats window.
- `scripts/compare_detectors.py`:
  - FPS and per-class counts per backend on images and videos.
  - `--eval` extracts only `images/test`, `labels/test` and `data.yaml` from `merged.zip`
    (needs `merged.done`) into `/content/datasets/merged_test/`. It checks that the zip's
    class names match `classes.yaml`, then computes mAP50 per class with its own
    COCO-style AP. Hybrid isn't one Ultralytics model, so `model.val()` can't score it.
- `notebooks/08_compare_detectors.ipynb`: runs the eval on a GPU and shows one table,
  class | COCO-only | openvocab | hybrid mAP50, with the rare classes highlighted. It also
  shows GPU FPS and counts, and saves JSON to `runs/compare_detectors/`.
- `.gitignore`: `models/*.ts` (MobileCLIP, 600 MB) and `models/openvocab/`.
  `requirements.txt`: the Ultralytics CLIP package.
- Measured on M1 CPU, macOS 14 (no MPS):
  - yolo11n: 52 ms at 640.
  - YOLOE-11s: 62 ms at 416, 75 ms at 480, 165 ms at 640.
  - ONNX Runtime was slower for YOLOE (~180-230 ms), so `runtime: pt` is the default.
  - Hybrid live with the debug window on a 720p video: 11.0-11.6 FPS (55-58 frames per
    5 s) with openvocab imgsz 416 and every 2nd frame. Frame-to-decision averaged
    86-91 ms, max 142 ms. The settings tried before were 480 every 2nd frame (10.0-10.6
    FPS) and 480 every 3rd frame (~12.3 FPS, but older hazard boxes).
  - Default backend unchanged: ~21 FPS, 64 ms.
- Prompts were tuned on 24 Wikimedia street photos (bollards, manholes, stairs, potholes,
  cones, utility poles):
  - Synonyms raised the bollard, pothole and stairs scores. "power pole" helped poles most.
  - Open-vocab hit 13/24 photos at 416.
  - In the live hybrid run, stairs, traffic cones, poles, potholes, bollards, manholes and
    barriers were all spoken.
- Tests: 140 pass.
  - New `tests/test_openvocab.py` (25, fake models only): vocabulary and hash, phrase to
    class mapping, per-class thresholds, prompt once then cache, YOLO-World fallback, ONNX
    export, NMS and merge, hybrid scheduling and reuse, the factory class split,
    `--backend`, and the spoken fallback.
  - New `tests/test_compare_detectors.py` (15): AP math, extracting only the test split,
    reuse, the class mismatch check, frames, the tables.
  - `test_detector.py` and `test_navigator.py` updated for the new config.
- Not tested or open:
  - The live webcam, pointed at photos on a screen. Only tested on a video of the photos;
    the camera's own CPU use may cost some FPS (`openvocab_every: 3` gives more margin).
  - The notebook in Colab: sections 2-4 ran locally on a fake `merged.zip`. No real mAP
    numbers yet.
  - The real YOLO-World fallback (only mocked; it downloads another ~340 MB CLIP).
  - Poles stay the weakest class (scores 0.1-0.26).
  - Box-height distance is poor for flat hazards (curb, pothole, manhole, crosswalk).
  - Crosswalk severity is 1, so a near crosswalk straight ahead is still announced as
    urgent.

## feat: merged dataset as one verified zip on Drive (2026-11-12)

- Why: the merged folder on Drive was incomplete. Drive filled up during the background
  upload, so only 20,924 train images arrived, with 0 val and 0 test. The notebook still
  said "copied 100%", and `check_det_dataset` passed only because it read Colab's local
  cache. The broken folder was deleted.
- `scripts/merge_datasets.py`, full merge:
  - Every image goes through the resize step. Images over 1280 px are resized, and PNGs
    and other formats become JPEG q90. Small JPEGs stay byte-identical. Every merged
    image is now `.jpg`.
  - Builds in `/content/merged_tmp`, then writes one file, `datasets/merged.zip`
    (ZIP_STORED), with `merge_report.json` and `review/` next to it.
  - Contents of the zip: `data.yaml` (with `path: /content/datasets/merged`), images,
    labels, the report and `md5_index.json`.
  - Checks free space before writing, needing the zip + report + review + 10%. If the
    upload replaces an old zip, that zip's size counts as free. The error message
    mentions Drive's Trash.
  - After writing:
    - `colab_sync()` runs `drive.flush_and_unmount()` then mounts again (outside Colab,
      `os.sync()`).
    - It verifies that the size on Drive equals the local size, the file count matches,
      and `testzip()` passes.
    - Only then does it write `merged.done` (zip bytes, files, images, free space left).
    - The old `merged.done` is removed before uploading.
  - Prints the raw sizes per folder, the images before/after resize, the zip size, the
    space check and the Drive space left. Image sizes are also saved in the report.
  - Includes `/content/raw` (rf_sidewalk and rf_barricade from section 2) together with
    Drive raw. A missing source folder fails with a hint about section 2.
- Append works on the zip:
  - It copies the zip to the local disk and unzips it, appends and re-maps (logic moved
    into `append_to_dataset()`), re-zips, then runs the same space check, upload,
    verification and `merged.done`.
  - Review images are added to the existing `review/`.
- `src/colab_data.py` `prepare_dataset()`:
  - Requires `merged.done`, copies the zip to `/content`, checks its size against
    `merged.done`, and unzips to `/content/datasets/merged/`.
  - Rewrites `data.yaml` with the local path, and compares the image count per split
    with `merge_report.json` next to the zip.
  - Runs `check_det_dataset` and returns the `data.yaml` path.
  - Reuses an unpack from the same zip in the same runtime.
- `notebooks/02_merge_datasets.ipynb`:
  - New settings `MERGED_ZIP` and `MAX_SIDE`. Shows Drive free space at setup.
  - The merge/append writes the zip, and section 4 is `prepare_dataset(MERGED_ZIP)`.
  - Review images are read from next to the zip.
  - The full merge also reads `LOCAL_RAW`.
  - The later "2.3 report" and training notebooks are not in the repo yet. They must use
    `prepare_dataset()` too.
- `CLAUDE.md`: a rule that the merged dataset is a verified zip loaded with
  `prepare_dataset()`.
- Tests: `tests/test_merge.py` was rewritten for the zip, with 40 merge and append tests.
  Added `tests/test_colab_data.py` with 5 tests. New cases:
  - Full merge:
    - zip, report and done written, ZIP_STORED, review kept outside the zip
    - resize, including a source from a second raw folder
    - a missing source folder fails
  - Drive safety:
    - full Drive stops before writing anything
    - a truncated upload or a corrupt byte (same size) never gets `merged.done`
  - Append:
    - Drive full leaves the zip and done unchanged
    - a truncated upload removes done
  - `prepare_dataset`:
    - unpacks, rewrites the path and runs Ultralytics' check
    - reuses an earlier unpack, requires `merged.done`
    - detects an image-count or zip-size mismatch
- Tested:
  - Full suite 96/96 passed. The merge and colab_data tests passed 3 more times.
  - Notebook 02 sections 2 → 3 → 4 were run locally with a fake Roboflow:
    - the download went to the local raw folder
    - the full merge read both raw folders and produced a verified zip
    - `prepare_dataset` succeeded
    - a re-run skipped the merge
- Not tested:
  - Colab:
    - `drive.flush_and_unmount()` and the remount in the middle of a run (it may ask to
      authorize again)
    - whether Drive's free-space figure is accurate
    - real sizes and timing
  - Whether a replaced zip still takes space as an old version or in the Trash.
    Verification would catch a failed upload.

## feat: notebook 02 downloads new sources before appending (2026-11-12)

- Why: every Colab notebook runs in its own runtime, so files that notebook 01 downloads
  to `/content/raw` are not visible to notebook 02.
- `scripts/download_datasets.py`: the download code from notebook 01, now shared by both
  notebooks.
  - Folder helpers: `is_done`, `skip_if_done`, `fresh_staging`, `publish` (staging →
    raw, then `.done`) and `record_failure`.
  - `colab_secret()` reads Colab Secrets.
  - Roboflow: `download_roboflow` keeps the latest-version fallback, and
    `download_roboflow_datasets` skips finished downloads and returns failures.
  - Class check: `read_class_names`, `mapping_problems`, `mappings_from_dir` and
    `check_mappings` print classes missing from the mapping files.
  - It logs with `logging` instead of `print`.
- `configs/datasets.yaml`: the Roboflow dataset list moved here from notebook 01's
  Settings cell, so both notebooks read one list.
- `notebooks/01_download_datasets.ipynb` works as before:
  - Setup now clones or updates the repo (like notebook 02) and imports the module.
  - The Roboflow cell is 3 calls. The class check reads the mapping files from the
    clone instead of GitHub URLs.
  - Open Images, COCO, Mendeley and Summary are unchanged, through the thin
    `fresh_staging` and `record_failure` wrappers.
  - Outputs of the 3 rewritten cells were cleared. Other cells and Colab outputs were
    kept.
- `notebooks/02_merge_datasets.ipynb`:
  - New setting `DOWNLOAD_NEW = ["rf_sidewalk", "rf_barricade"]` (`[]` = skip).
  - New section "2. Download new sources" before the merge/append:
    - Installs roboflow and reads `ROBOFLOW_API_KEY` from Secrets.
    - Downloads only those sources to `LOCAL_RAW` (`/content/raw`), skipping any with
      `.done`.
    - Prints classes missing from their mapping files, and stops if a download failed.
  - The append then finds the files in the same runtime.
- `tests/test_download.py`: 14 tests with a fake Roboflow. They cover:
  - downloads marked done, skipping finished ones, replacing partial ones
  - latest-version fallback, failures recorded while the rest continue, missing key
  - `select_datasets`, and the real dataset list (the 2 new sources, a mapping file for
    each name)
  - class-name reading, mapping problems and the class-check log
  - both notebooks compile, no copied download code, and notebook 02 downloads before
    the append
- Tested:
  - Full suite 87/87 passed.
  - Notebook 02's download, append and check cells were run together locally with a fake
    roboflow module and Colab secret. The download went to the local raw folder, a
    re-run skipped it (`.done`), and the append found it and passed
    `check_det_dataset`.
- Not tested:
  - Real Roboflow downloads in Colab.
  - Notebook 01's new setup cell (repo clone) in Colab.

## feat: append mode for merging new sources (2026-11-12)

- `scripts/merge_datasets.py`: `append()` / `--append SOURCE ...` adds sources to the
  existing `datasets/merged/` without rebuilding it.
  - Requires `.done`, `data.yaml` and `merge_report.json`, and fails if
    `configs/classes.yaml` differs from the merged `data.yaml`.
  - Uses the same mapping and validation rules. Sources already in the report are skipped,
    so re-runs are safe.
  - De-duplicates by md5 against `md5_index.json` (source-file md5 → `split/name`).
    - Full merges now write this index.
    - For older merges it's built once, by hashing the merged images.
  - Grouped 75/15/10 split of the new images only, with seed 42.
  - New images are stored as JPEG with the longer side at most `max_side` (1280),
    quality 90. JPEGs that are already small are copied byte for byte.
  - Builds in a local work dir, then checks free space on the target with a 10% margin.
    If it doesn't fit, it stops before writing anything.
  - Changed mappings: any merged source whose mapping changed has its existing labels
    re-mapped from its raw labels in the same run. Only changed label files are
    rewritten.
    - Change detection uses the `mapping` now recorded per source in the report.
    - For older reports, it falls back to "a dropped class is now mapped". This covers
      `pedestrian-crossing: drop` → `crosswalk`.
  - Updates `merge_report.json`: new and re-mapped source entries, recomputed totals and
    low-count classes, and an `appends` log. Prints the class table.
    - An `.append_in_progress` marker covers interrupted runs.
  - `--raw` now takes several folders, searched in order (`/content/raw` before Drive
    raw).
  - New mapping option `missing: warn`: mapped classes the source lacks only warn.
  - Refactored the report code into `source_report()` and `finalize_report()`, and
    folder copying into `copy_files()`.
- New mapping files:
  - `rf_barricade`: Barricade → barrier.
  - `rf_sidewalk`: barricade → barrier, plus 13 classes with the same names.
    `unlisted: drop`, `missing: warn`. Its class names weren't checked against the
    real `data.yaml`.
- `notebooks/01_download_datasets.ipynb`:
  - Added rf_sidewalk (scottsdale/sidewalk-otzhb v1) and rf_barricade
    (myplace-65bsf/barricade-detection-kaaty v10).
  - `RAW_TARGET = "drive" | "local"` (local = `/content/raw`), and an `ONLY` filter for
    all sections.
  - After the Roboflow downloads, it compares each dataset's classes with its mapping
    file on GitHub and prints unmapped classes and missing mapped classes.
  - These were text edits, so Colab's cells, ids and outputs are kept.
- `notebooks/02_merge_datasets.ipynb`:
  - New settings: `APPEND`, `NEW_SOURCES`, `LOCAL_RAW`, `APPEND_WORK` and `MAX_SIDE`.
  - The merge cell calls `append()` or `merge()`.
  - The review notes explain how to switch pedestrian-crossing to crosswalk.
- `tests/test_merge.py`: 12 append tests, using `tests/assets/merge/new_raw/src_c` (a
  2000×1000 PNG augmented pair, a small JPEG, a copy of a merged image, an unlisted
  class). They cover:
  - totals checked against the label files on disk
  - resizing, and the small JPEG unchanged
  - the pair kept in one split, the md5 index, and building the index for older merges
  - re-run skip, full-Drive stop with nothing written, unmapped class, no merge yet
  - re-mapping a changed source, and changed-class detection for old reports
- Tested:
  - Full suite 73/73 passed.
  - Notebook 01's class check was run on fake downloads. It caught `fire hydrant` vs
    `fire_hydrant` and an unmapped class.
  - Notebook 02's merge and append cells were run locally on the fake data, and
    `check_det_dataset` passed after both.
- Not tested:
  - Append on the real Drive dataset in Colab: timing of the one-time md5 index, and
    whether Drive's free-space figure is accurate.
  - Re-mapping rf_crosswalk on real data.
  - The real rf_sidewalk and rf_barricade class names.
- Open:
  - With the new mapping files, a full rebuild needs rf_sidewalk and rf_barricade in a
    raw folder.

## feat: merge raw datasets into one YOLO dataset (2026-11-10)

- `configs/classes.yaml`: the 20 target classes in ID order, person (0) to fire_hydrant (19).
- `configs/mappings/<source>.yaml`: one file per source, mapping each source class to a
  target class or `drop`.
  - Sources: coco_subset, openimages, mendeley, rf_bollard, rf_cone, rf_crosswalk,
    rf_manhole, rf_pole_curb_stairs, rf_pothole, rf_scooter_bollard and rf_stairs.
  - rf_crosswalk: `pedestrian-crossing` and `2` are dropped. `pedestrian-crossing` is
    marked for review.
  - `mapillary.yaml` (instance masks → boxes, min 12 px, other labels dropped) is used
    only if `raw/mapillary/` exists. Its label names are unverified guesses at Vistas
    v2.0.
- `scripts/merge_datasets.py` (`merge()` and a CLI):
  - Prints all mappings first, then scans and validates every source before copying
    anything.
  - Fails with a clear message on:
    - a source class that isn't mapped
    - a mapped class the source doesn't have (with similar-name suggestions)
    - a raw folder without a mapping file
    - a missing source folder
    - more than 50% of a source's images without a label file (an unknown layout)
  - Builds on the local disk (`--work`):
    - Copies the images while hashing them, and removes exact duplicates by md5 (the
      first source in mapping order wins).
    - Prefixes filenames with the source name, and rewrites the labels with target IDs.
    - Converts polygon labels to boxes. Invalid lines are skipped and counted.
  - Grouped 75/15/10 split per source, seed 42:
    - Roboflow `.rf.` copies stay together.
    - Groups in a source's own test split go to test, and the rest fill test up to
      10%, then val.
  - Writes `data.yaml` (absolute `--out` path), `merge_report.json` (images and boxes per
    split, per class and per source, plus duplicates, dropped boxes and invalid boxes),
    and 16 review images per review class in `review/<class>/` (red = review boxes,
    green = kept boxes).
  - Prints a table of boxes per class per split, and flags classes under 300 train boxes.
  - Copies to `--out` with threads and tqdm, then writes `.done`. Re-runs skip unless
    forced.
  - Only deletes folders that carry its own marker.
- `notebooks/02_merge_datasets.ipynb`:
  - Mounts Drive and clones or updates the repo on `develop`.
  - Merges `datasets/raw` into `datasets/merged` through `/content/merged_tmp`.
  - Runs `check_det_dataset`, then shows the review images.
- `notebooks/01_download_datasets.ipynb`: removed `rf_sidewalk_obstacles`, whose download
  failed with no versions.
- `requirements.txt`: `tqdm>=4.66`.
- `tests/test_merge.py`: 20 tests on two fake sources in `tests/assets/merge/`. They
  cover:
  - the end-to-end output and the re-run skip
  - prefixes, duplicate removal and remapped IDs
  - polygons, the augmented pair in one split, and the own test split kept
  - report counts and review images
  - Ultralytics `check_det_dataset`
  - the failure cases: unmapped class, mapping typo, unmapped folder, unknown label
    layout, and a folder that isn't the script's own
  - unit tests for grouping, label parsing, the split ratios and seed, Mapillary
    mask-to-box conversion, and the real mapping files
- Tested:
  - Full suite 62/62 passed.
  - The CLI and the notebook's merge and check cells were run locally on the fake data.
- Not tested:
  - The merge on the real Drive datasets in Colab, including timing and the Mendeley
    folder layout.
  - Mapillary on real data.
- Open:
  - `notebooks/00_colab_template.ipynb` still doesn't exist.
  - Decide on `pedestrian-crossing` after looking at the review images.
  - Sources don't label each other's classes (for example, people in the cone images),
    so the training data has unlabeled objects.

## feat: Colab notebook to download raw datasets (2026-11-09)

- `notebooks/01_download_datasets.ipynb` fills `MyDrive/clear_vision/datasets/raw/<name>/`.
  - A Settings cell at the top holds `ROBOFLOW_DATASETS`, the Open Images and COCO class
    lists and the sample limits.
  - 1. Setup: mounts Drive, installs roboflow and fiftyone, and reads `ROBOFLOW_API_KEY`
    from Colab Secrets. If the key is missing it warns and skips the Roboflow datasets.
  - 2. Roboflow: 9 datasets in YOLOv8 format. If a version fails, it uses the project's
    latest version and prints which one.
  - 3. Open Images V7: 8 classes, loaded one class at a time (up to 1500 per class, per
    split), merged without duplicate images. It exports only those classes' boxes to
    YOLOv5 format (train and val).
  - 4. COCO-2017 subset: 10 classes, 5000 train and 1000 validation images, exported to
    YOLO. The validation split was added so the export has a val set.
  - 5. Mendeley: unzips `raw/mendeley/*.zip` into `raw/mendeley/<zip name>/`.
  - 6. Summary table: status, images, label files and class names (from `data.yaml`,
    `dataset.yaml` or a classes file), plus a list of failures.
  - Re-running is safe:
    - Each download goes to the Colab disk first and is then copied to Drive. A `.done`
      marker is written last.
    - Folders with `.done` are skipped. Folders without it are partial and downloaded
      again.
    - A failure in one dataset is recorded and doesn't stop the others.
- `notebooks/00_colab_template.ipynb` wasn't in the repo, so the header follows the
  existing `clear-vision/Clear_Vision.ipynb`: a Colab badge pointing at `develop`.
- Tested:
  - The notebook passes `nbformat.validate`, and every code cell compiles.
  - The Mendeley and Summary cells were run locally on a fake Drive folder, with Colab
    stubbed out. Unzip, skip-on-rerun, and the done / PARTIAL / FAILED rows all worked.
- Not tested:
  - The Roboflow, Open Images and COCO downloads. These need Colab and the API key.
  - Running all cells in Colab end to end.

## feat: main app wiring camera, detector, navigator and speaker (2026-11-08)

- `src/main.py` (`python -m src.main`) connects Camera → Detector → `choose_message` →
  Speaker.
  - CLI: `--source` (default from config), `--config`, `--no-window`, `--mute`.
  - Debug window:
    - Boxes with class, confidence, direction and proximity: red for urgent
      (ahead + near), orange for risk ≥ threshold, green otherwise.
    - Lines marking the zones.
    - FPS, capture age, inference ms and total ms at the top, and the last message at
      the bottom.
  - Speech:
    - "Clear Vision started" on launch.
    - "Camera not available" or "Camera lost" once, then "Camera back".
    - "Detection model failed…" out loud, then exit.
  - `CameraSupervisor` reopens a dead camera every `camera_retry_s` on a background
    thread. It also stops hung cameras off the main loop.
  - Normal messages are only sent when the speaker is idle, so they don't pile up.
    Urgent always goes through. `last_spoken` is set when a message is sent.
  - `logs/session_<datetime>.csv`: one row per message with time, elapsed, priority,
    key, text, frame-to-decision ms and muted.
  - Graceful shutdown on `q`, Ctrl+C or video end. FPS and timings are logged every
    `stats_interval_s`.
- `src/speaker.py`: added `is_busy`, true while speaking or while a message is pending.
- `configs/default.yaml`: new `app:` section with `camera_retry_s`, `show_message_s`
  and `stats_interval_s`.
- `.gitignore`: `logs/`.
- Tested:
  - Synthetic 6 s zoom video from the COCO street image, run headless and muted:
    - 23.7 FPS, frame-to-decision avg 67.5 ms (max 113 ms), inference about 46 ms.
    - 8 messages in 6 s with no per-frame repeats. Urgent repeats only after the 1.5 s
      cooldown.
  - Missing camera: "Camera not available", a retry every 2 s, clean exit on SIGINT.
  - Full suite 42/42 passed.
- Not tested:
  - The live webcam: walking toward a chair or person, with audio.
- Still open:
  - The `say` start-up delay of about 0.6 s.
  - The first frame at a new shape takes about 0.25 s.
  - The tracker isn't built yet.

## feat: non-blocking speaker with priorities (2026-11-08)

- `src/speaker.py`: `Speaker` runs TTS on its own thread. `say()` never blocks.
  - There's at most one pending message:
    - Urgent replaces pending and interrupts current speech by killing the `say`
      process or calling `engine.stop()`.
    - A normal message replaces an older pending normal (newest wins). It's dropped if
      an urgent message is pending, and it never interrupts.
  - `say()` accepts a navigator `Message` or a plain string with a priority.
  - Also has `start()`, `stop()`, `with Speaker(...)`, `is_speaking`,
    `wait_until_idle()` and `Speaker.from_config()`.
  - Backends: `SayBackend` (`say -r <rate>`) and `Pyttsx3Backend`.
    - `auto` = `say` on macOS, pyttsx3 elsewhere.
    - If pyttsx3 fails at runtime on macOS, the speaker switches to `say`.
    - If every backend fails, it logs a critical error.
  - Why `say` on macOS: tested pyttsx3's nsss driver on a worker thread.
    `runAndWait()` returns in about 0.05 s while speech continues, so the end of an
    utterance can't be detected and queued messages would overlap.
- `configs/default.yaml`: new `speaker:` section with `rate: 220` and `backend: auto`.
- `scripts/speaker_test.py`: demonstrates out loud:
  - urgent interrupts normal
  - only the newest normal is kept
  - normal dropped while urgent is pending
- `tests/test_speaker.py`: 8 silent tests of the queue rules, using a fake backend.
- Removed the empty `src/spiker.py`, which `speaker.py` replaces.
- Tested:
  - The demo interrupted the normal sentence 9 ms after the urgent message was queued.
  - Speaker tests passed on 5 runs in a row. Full suite 42/42 passed.
- Open:
  - Each `say` call takes about 0.7 s to start, rendering "Person, left" in 0.68 s.
    That alone breaks the 300 ms budget. Fix: a persistent engine or pre-rendered
    phrases.
  - `main.py` isn't wired yet, so there's no real-time test with real objects.
  - The tracker is missing.
  - The COCO model can't see poles, curbs, stairs or holes.

## feat: navigator logic and navigator tests (2026-11-08)

- `src/navigator.py` contains pure functions: no OpenCV, no audio, no camera.
  - `direction(det)`: left, ahead or right, by `center_x` against `left_boundary` and
    `right_boundary` (0.333 and 0.667).
  - `proximity(det)`: by box height. near above 0.5, mid from 0.25 to 0.5, far below.
  - `severity(class_name)`: from the config table. Unknown classes get
    `default_severity`.
  - `risk(det)`: proximity weight × direction weight × severity.
    - Proximity weights: near 1.0, mid 0.6, far 0.3.
    - Direction: ahead ×1.5, side ×1.0.
  - `choose_message(detections, now, last_spoken)`: returns the highest-risk object
    above `risk_threshold` (1.0) as a `Message(text, priority, key)`, or None.
    - The phrase looks like "Car ahead, close" or "Person, left". It's urgent when the
      object is ahead and near.
    - The cooldown is per class and direction: 4 s normally, `urgent_cooldown` 1.5 s
      for urgent warnings.
    - If the top object is on cooldown, the next one is announced.
    - It doesn't modify `last_spoken`. The caller records the time after speaking.
  - `NavigatorConfig.from_dict()` builds the config from the `navigator:` section.
- `src/detection.py`: `Detection` moved here, so the navigator and its tests don't
  import torch, Ultralytics or OpenCV. `src.detector` still re-exports it.
- `configs/default.yaml`: new `navigator:` section with every threshold, weight,
  severity and cooldown.
- `tests/test_navigator.py`: 30 cases covering:
  - directions and distances, including boundary values
  - severity and risk ordering
  - no message, and below the threshold
  - urgent and normal phrases
  - most dangerous object chosen
  - cooldown: suppress, per key, fallback, shorter urgent cooldown, history unchanged
  - no cv2, torch or Ultralytics imported
- Tested: navigator tests 30/30 in 0.08 s with no camera. Full suite 34/34 passed.
- Open:
  - Not connected to the live camera yet. That needs `main.py`, or a live test script.
  - `speaker.py` and the tracker are still missing.

## feat: YOLO detector, config file and detector tests (2026-11-08)

- `src/detector.py`:
  - `Detection(class_name, confidence, x1, y1, x2, y2, track_id=None)`. Coordinates are
    normalized to 0–1, with the `center_x`, `width`, `height` and `area` properties.
  - `Detector(model_path, conf, allowed_classes, imgsz, device, cpu_threads)` and
    `Detector.from_config()`.
    - It uses Ultralytics YOLO, and `yolo11n.pt` auto-downloads into `models/`.
    - It warms up when created.
    - The class filter is applied again after inference, because exported ONNX models
      ignore `classes=`.
  - `device: auto` uses MPS only on macOS 15+. On macOS 14.5 with torch 2.15, MPS
    crashes the whole process with a native error that Python can't catch.
- `configs/default.yaml`: `camera:` and `detector:` sections, including the 12
  `allowed_classes` and `cpu_threads: 4`.
- Tests:
  - `tests/test_detector.py` has 4 tests: Detection math, finds a person or car, only
    allowed classes and normalized boxes, and timing printed for a 640 px frame.
  - Sample image: COCO val2017 #000000011197, a street with cars, people and a bicycle.
  - `pytest.ini` lets plain `pytest` find `src`.
- `requirements.txt` filled in. `.gitignore` now ignores `models/*.pt`, `models/*.onnx`
  and `.pytest_cache/`.
- Installed PyYAML and pytest. Ultralytics also installed onnx and onnxslim by itself
  during the ONNX benchmark.
- Benchmarks on the M1 CPU, per frame:
  - 8 threads: about 58 ms. 4 threads: about 55 ms. 1280×720 frames: about 44 ms.
  - ONNX through Ultralytics: no faster (CPU provider only), and 79 ms on 720p frames
    because of the fixed 640×640 input.
- Tested: 4/4 tests passed, with a median of 56.1 ms per 640 px frame.
- Open:
  - No live webcam detection viewer yet.
  - A CoreML / ONNX Runtime speedup is still to do.
  - MPS needs macOS 15.

## feat: camera capture module + camera test script (2026-11-08)

- `src/camera.py`: added the `Camera(source, width, height, stale_timeout=2.0)` class.
  - `source` can be a webcam index, a video file path or a stream URL. A numeric string
    like `"0"` becomes a webcam index.
  - A background thread reads frames and keeps only the latest one, so there's no lag
    from a backlog of old frames.
  - `start()`, `read()`, `stop()`, and `with Camera(...)` support.
  - `read()` returns `(frame, timestamp)`, or `(None, None)` before the first frame.
    The timestamp is `time.monotonic()` at capture, for latency measurement.
  - `healthy` turns False after 2 s without a new frame. It's computed on access, so
    it still works when the backend's `read()` hangs.
  - Video files play at the file's real FPS. `ended` becomes True when playback finishes.
  - If the source can't be opened, `start()` raises a `RuntimeError` with a clear message.
- `scripts/camera_test.py`: shows the live feed with FPS and OK / NO SIGNAL drawn on
  screen. `q` quits. It accepts a webcam index, a video path or a URL.
- Tested headless with a synthetic 3 s, 30 FPS clip:
  - It played in 3.00 s.
  - `healthy` went False 2 s after the clip ended.
  - A missing file raised the error.
- Not tested yet: the webcam at 25+ FPS.

## e6ab82d — feat: CLAUDE.md added (2026-11-08)

- Created `CLAUDE.md` as the permanent project context:
  - goal and the 300 ms latency budget
  - the Colab / local workflow split and Google Drive layout
  - tech stack and `src/` architecture
  - coding rules, git habits and commands
- Noted gaps between CLAUDE.md and the repo, still open:
  - `src/spiker.py` should be `speaker.py`.
  - The notebook is in `clear-vision/`, not `notebooks/`.
  - There's no tracker module or `configs/default.yaml` yet.
  - `requirements.txt` is empty.
  - There's no `src/__init__.py`.

## 5a3ac9d — Set up project structure and Python 3.11 environment (2026-11-08)

- Linked the local folder to https://github.com/Zikirullo/clear-vision.git.
- Created a Python 3.11 venv in `.venv/` and installed:
  - ultralytics, opencv-python, pyttsx3, numpy, onnxruntime
  - torch and torchvision, which ultralytics pulls in
- Added `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `.ipynb_checkpoints/`.
- Moved the Colab notebook to `clear-vision/Clear_Vision.ipynb`. It's byte-identical to
  GitHub's original.
- Removed the nested `.git` folders in `src/` and `clear-vision/`. They were moved to a
  temporary backup, and nothing in them was missing from history.
- Added the empty `src/*.py` skeleton files and `requirements.txt`.
- Branches:
  - `develop` (GitHub default) and `master` both point to this commit.
  - Old `main` is deleted.
  - History was rebuilt linear, then force-pushed with approval.
- Git config:
  - `origin/HEAD` removed and `remote.origin.followRemoteHEAD never` set.
  - `pull.rebase true` keeps history linear when Colab and VS Code both push to
    `develop`.

## e770269 — BBR: Initial Setup (2026-11-03)

- Added the Colab notebook `Clear_Vision.ipynb` from Google Colab.

## 5c352f6 — Initial commit (2026-11-03)

- Repo created on GitHub with `README.md`.
