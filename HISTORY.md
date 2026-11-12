# History

What was done before each commit, newest first. Add a new entry before every commit
(see "History log" in CLAUDE.md).

---

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
