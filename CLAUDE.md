# Clear Vision

AI navigation assistant for blind and visually impaired people. A chest- or
glasses-mounted camera detects obstacles while the user walks and warns them with
short spoken messages (vibration later).

**Latency budget:** frame captured → audio starts in **under 300 ms**, end to end.

## Workflow split

**Google Colab (T4 GPU):** data exploration, training, fine-tuning, model export.
- Notebooks live in `notebooks/` and are opened in Colab from GitHub.
- Datasets and runs live on Google Drive in `MyDrive/clear_vision/`:
  `datasets/`, `runs/`, `exports/`.
- The merged dataset is one verified `datasets/merged.zip` (notebook 02). Notebooks load it
  with `src/colab_data.py` `prepare_dataset()`, never from a folder of files on Drive.
- Notebook 03 writes `train_balanced.txt` + `data_balanced.yaml` next to the zip; training
  applies them after `prepare_dataset()` with `apply_balanced()`.
- New sources are added with notebook 02's append mode (focus only, pHash leakage guard).
  rf_safewalk is held out whole as the `external_test` split, never trained on. Notebook 10
  (`scripts/safety_report.py`) reports `test`, `external_test` and the 24 real street photos
  in `tests/assets/street_photos/` separately. Judge a model by the last two.

**Local (MacBook, VS Code):** the real-time app in `src/`. This covers camera
capture, ONNX Runtime inference, navigation logic, audio output and latency tests.

**Phone (`mobile_flutter/`, Flutter, iPhone first, Android too):** the same rules ported to
Dart in `mobile_flutter/lib/navigation/`, detection by the `ultralytics_yolo` plugin. Its
tests replay `tests/golden/nav_cases.json` (`cd mobile_flutter && flutter test`). After changing
the rules or config, also run `python scripts/sync_mobile_assets.py --config-only` (copies
`nav_config.json` into the app) and port the change to Dart. `--run v2` swaps the model.
Setup and signing: `mobile_flutter/README.md`.

## Git

- Work on `develop`, which is GitHub's default branch. `master` is the stable branch.
- Colab and VS Code both push to `develop`. In VS Code, pull before you start and
  push when you finish. `pull.rebase` is on, so history stays linear.
- Never edit the same notebook in Colab and VS Code at the same time.
- Never commit `.venv/`, datasets, runs or large model weights.

## History log (do this before every commit)

Before each commit, add an entry at the top of `HISTORY.md`. Use the commit message
as the heading. List what changed and why, how it was tested, and anything not tested
or left open. Stage `HISTORY.md` in the same commit. Never rewrite older entries.

## Tech

Python 3.11 (venv in `.venv/`), OpenCV, Ultralytics YOLO, ONNX Runtime,
pyttsx3 (fallback: macOS `say`), NumPy, PyYAML, pytest.

## Architecture (`src/`)

```
camera.py → detector.py → tracker → navigator.py → speaker.py
                         main.py wires it all together
```

- `camera.py`: threaded capture that keeps only the latest frame and drops stale ones.
- `detector.py`: model inference. Backends (`detector.backend` or `--backend`):
  `ultralytics` and `onnx` (COCO yolo11n), `openvocab` (YOLOE prompted from
  `configs/prompts.yaml`, no training), `finetuned` (`models/budget_v1/best.pt`, notebook 05),
  and `hybrid` (the default). Hybrid uses the fine-tuned model if it exists, either alone
  (`finetuned.only`) or per class from `per_class_choice.yaml`; otherwise it uses yolo11n +
  YOLOE. All return `Detection`s with `configs/classes.yaml` names.
- tracker: keeps object IDs stable across frames.
- `navigator.py`: works out direction, distance and a risk score for each object,
  then decides what to say.
  - Speech filter: per-class speak thresholds (separate from the detection thresholds),
    3-of-5-frame confirmation, a horizon/size zone, cooldowns, and hazards-only.
  - `planner.py` holds the frame history and last-spoken times for `main.py` and
    `scripts/replay.py`.
  - The rules are specified in `docs/navigation_spec.md` for the phone app. After changing
    them or the `navigator:` config, run `scripts/export_nav_config.py` (writes
    `models/nav_config.json`) and `scripts/make_nav_golden.py` (writes
    `tests/golden/nav_cases.json`), then review both diffs.
- `speaker.py`: non-blocking TTS queue with priorities. Urgent warnings jump the queue.
- `main.py`: wires the pipeline together, with a debug window and an FPS readout.

All tunable numbers live in `configs/default.yaml`, including thresholds, distances,
cooldowns, priorities and model paths. Never hard-code them.

## Coding rules

- Type hints everywhere.
- Keep `navigator.py` as small pure functions, so it can be unit-tested with pytest
  without a camera or model.
- Log with the `logging` module, not `print`.
- No blocking calls in the main loop. I/O and TTS run in their own threads or queues.
- Safety first:
  - When unsure, warn.
  - If the camera or model fails, say so out loud. Never fail silently.
- Measure latency whenever you change the pipeline. A change that breaks the 300 ms
  budget is a regression.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main           # run the app
python -m src.main --backend hybrid   # zero-training hazard detection
python scripts/compare_detectors.py photos/   # FPS + counts per backend
python scripts/replay.py clip.mp4 [--no-filter]   # every message the app would speak, with times
pytest                       # run tests
cd mobile_flutter && flutter test   # Dart port: golden cases + app logic
```

Core ML (iPhone) export runs on the Mac in its own environment, `.venv-export` (coremltools
needs numpy < 2.4 and torch 2.7; see `requirements-export.txt`):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-export
.venv-export/bin/python -m pip install --only-binary=:all: -r requirements-export.txt
.venv-export/bin/python scripts/export_coreml.py --run budget_v1 --images tests/assets/street_photos
```
