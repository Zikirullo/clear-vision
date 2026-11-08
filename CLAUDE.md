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

**Local (MacBook, VS Code):** the real-time app in `src/`. This covers camera
capture, ONNX Runtime inference, navigation logic, audio output and latency tests.

## Git

- Work on `develop`, which is GitHub's default branch. `master` is the stable branch.
- Colab and VS Code both push to `develop`. In VS Code, pull before you start and
  push when you finish. `pull.rebase` is on, so history stays linear.
- Never edit the same notebook in Colab and VS Code at the same time.
- Never commit `.venv/`, datasets, runs or large model weights.

## Tech

Python 3.11 (venv in `.venv/`), OpenCV, Ultralytics YOLO, ONNX Runtime,
pyttsx3 (fallback: macOS `say`), NumPy, PyYAML, pytest.

## Architecture (`src/`)

```
camera.py → detector.py → tracker → navigator.py → speaker.py
                         main.py wires it all together
```

- `camera.py`: threaded capture that keeps only the latest frame and drops stale ones.
- `detector.py`: model inference (ONNX Runtime locally).
- tracker: keeps object IDs stable across frames.
- `navigator.py`: works out direction, distance and a risk score for each object,
  then decides what to say.
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
pytest                       # run tests
```
