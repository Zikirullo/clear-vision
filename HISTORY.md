# History

What was done before each commit, newest first. Add a new entry before every commit
(see "History log" in CLAUDE.md).

---

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
