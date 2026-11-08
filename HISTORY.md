# History

What was done before each commit, newest first. Add a new entry before every commit
(see "History log" in CLAUDE.md).

---

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
