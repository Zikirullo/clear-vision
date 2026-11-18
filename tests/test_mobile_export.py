"""Tests for scripts/mobile_export.py (no export, no model)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

import scripts.mobile_export as me

ROOT = Path(__file__).resolve().parent.parent
CLASSES = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["names"]


def test_read_imgsz_from_metrics(tmp_path: Path) -> None:
    assert me.read_imgsz(ROOT / "models" / "budget_v1" / "metrics.json") == 512  # the committed budget_v1 run
    (tmp_path / "m.json").write_text(json.dumps({"settings": {"imgsz": 640}}))
    assert me.read_imgsz(tmp_path / "m.json") == 640
    assert me.read_imgsz(tmp_path / "missing.json") == 512
    (tmp_path / "empty.json").write_text("{}")
    assert me.read_imgsz(tmp_path / "empty.json", default=320) == 320


def test_calibration_yaml_uses_train_images_as_val(tmp_path: Path) -> None:
    train = tmp_path / "images" / "train"
    train.mkdir(parents=True)
    for i in range(250):
        (train / f"img{i:03d}.jpg").write_bytes(b"x")
    data = yaml.safe_load(me.calibration_yaml(tmp_path, CLASSES).read_text())
    lines = (tmp_path / "calib_train.txt").read_text().splitlines()
    assert len(lines) == 200 == len(set(lines)) and all(ln.startswith("./images/train/") for ln in lines)
    assert data["val"] == data["train"] == "calib_train.txt" and data["path"] == str(tmp_path.absolute())
    assert list(data["names"].values()) == CLASSES
    assert me.calibration_yaml(tmp_path, CLASSES) and (tmp_path / "calib_train.txt").read_text().splitlines() == lines
    small = tmp_path / "small"
    (small / "images" / "train").mkdir(parents=True)
    (small / "images" / "train" / "a.jpg").write_bytes(b"x")
    me.calibration_yaml(small, CLASSES)
    assert (small / "calib_train.txt").read_text() == "./images/train/a.jpg\n"


def test_thresholds_from_config() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    default, thr = me.thresholds(cfg, CLASSES)
    assert default == 0.25 and list(thr) == CLASSES
    assert {c for c, v in thr.items() if v == 0.15} == {"curb", "stairs", "pothole", "pole", "manhole"}
    assert all(v == 0.25 for c, v in thr.items() if c not in me.CRITICAL_CLASSES)


def test_side_files(tmp_path: Path) -> None:
    me.write_side_files(tmp_path / "mobile", CLASSES, 0.25, {"curb": 0.15, "car": 0.25}, "budget_v1", 512)
    assert (tmp_path / "mobile" / "labels.txt").read_text().splitlines() == CLASSES
    data = json.loads((tmp_path / "mobile" / "thresholds.json").read_text())
    assert data["run"] == "budget_v1" and data["imgsz"] == 512 and data["default"] == 0.25
    assert data["classes"] == {"curb": 0.15, "car": 0.25}


def test_zip_mlpackage_keeps_package_folder_on_top(tmp_path: Path) -> None:
    pkg = tmp_path / "clear_vision_budget_v1.mlpackage"
    (pkg / "Data" / "com.apple.CoreML").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}")
    (pkg / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"m")
    z = me.zip_mlpackage(pkg, tmp_path / "clear_vision_budget_v1.mlpackage.zip")
    with zipfile.ZipFile(z) as zf:
        assert sorted(zf.namelist()) == ["clear_vision_budget_v1.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                                         "clear_vision_budget_v1.mlpackage/Manifest.json"]
    with pytest.raises(FileNotFoundError):
        me.zip_mlpackage(tmp_path / "missing.mlpackage", tmp_path / "x.zip")


def test_tflite_metadata_appended_zip(tmp_path: Path) -> None:
    f = tmp_path / "m.tflite"
    f.write_bytes(b"TFL3" + b"\0" * 100)  # a flatbuffer, then Ultralytics appends a zip
    assert me.tflite_metadata(f) is None
    with zipfile.ZipFile(f, "a") as zf:
        zf.writestr("metadata.json", json.dumps({"task": "detect", "names": {"0": "person", "1": "car"}, "imgsz": [512, 512]}))
    meta = me.tflite_metadata(f)
    assert meta["task"] == "detect" and me.names_from_metadata(meta) == ["person", "car"]


def test_names_from_metadata_flavours() -> None:
    assert me.names_from_metadata({"names": "{0: 'person', 1: 'car'}"}) == ["person", "car"]  # Core ML stores str(dict)
    assert me.names_from_metadata({"names": ["a", "b"]}) == ["a", "b"]
    assert me.names_from_metadata({}) is None and me.names_from_metadata(None) is None


def test_result_rows_and_sizes(tmp_path: Path) -> None:
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"x" * 1_000_000)
    pkg = tmp_path / "m.mlpackage"
    pkg.mkdir()
    (pkg / "w.bin").write_bytes(b"x" * 500_000)
    per_class = {c: 0.5 for c in me.CRITICAL_CLASSES}
    base = me.result_row("PyTorch", pt, per_class, 0.70, 0.70, 1.0)
    coreml = me.result_row("Core ML FP16", pkg, None, None, 0.70, 1.0)
    fp16 = me.result_row("TFLite FP16", pt, per_class, 0.686, 0.70, 1.0)
    assert base["mAP50 loss %"] == 0.0 and base["x .pt size"] == 1.0
    assert coreml["size MB"] == 0.5 and coreml["mAP50"] is None and coreml["recall curb"] is None
    assert fp16["mAP50 loss %"] == 2.0
    table = me.format_rows([base, fp16, coreml])
    assert table.splitlines()[0].startswith("model") and "Core ML FP16" in table and " - " in table


def test_copy_artifact_replaces(tmp_path: Path) -> None:
    src = tmp_path / "a.mlpackage"
    src.mkdir()
    (src / "f").write_text("new")
    dest = tmp_path / "out" / "a.mlpackage"
    dest.mkdir(parents=True)
    (dest / "old").write_text("old")
    me.copy_artifact(src, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["f"]
    f = tmp_path / "x.tflite"
    f.write_text("1")
    me.copy_artifact(f, tmp_path / "out" / "x.tflite")
    assert (tmp_path / "out" / "x.tflite").read_text() == "1"


def test_command_line_passes_export_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    seen = {}

    def fake_export(weights: Path, fmt: str, out: Path, imgsz: int, **kwargs: object) -> Path:
        seen.update(weights=weights, fmt=fmt, imgsz=imgsz, **kwargs)
        out.write_bytes(b"x" * 2_000_000)
        return out

    monkeypatch.setattr(me, "export_one", fake_export)
    out = tmp_path / "m.mlpackage"
    assert me.main(["export", "best.pt", "coreml", str(out), "nms=true", "quantize=16", "conf=0.15",
                    "data=/x/data_calib.yaml", "--imgsz", "512"]) == 0
    assert seen == {"weights": Path("best.pt"), "fmt": "coreml", "imgsz": 512, "nms": True, "quantize": 16,
                    "conf": 0.15, "data": "/x/data_calib.yaml"}
    assert capsys.readouterr().out.strip().splitlines()[-1] == f'RESULT {{"path": "{out}", "size_mb": 2.0}}'
    assert me.main(["export", "best.pt", "litert", str(out), "nms=null", "--imgsz", "512"]) == 0
    assert seen["nms"] is None


def test_command_line_skips_core_ml_outside_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    import platform

    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": {0: "person"}}))
    (tmp_path / "thr.json").write_text(json.dumps({"classes": {"person": 0.25}}))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert me.main(["evaluate", str(tmp_path / "m.mlpackage"), str(tmp_path / "data.yaml"), "--imgsz", "512",
                    "--thresholds", str(tmp_path / "thr.json")]) == 0
    assert "Core ML runs only on macOS" in capsys.readouterr().out


def test_decode_handles_both_coordinate_conventions() -> None:
    import numpy as np

    raw = np.zeros((1, 6, 10), np.float32)  # 4 box rows + 2 classes, 10 anchors (real models: 5376)
    raw[0, :4, 0] = [100, 100, 40, 40]     # pixel coordinates (TensorFlow route)
    raw[0, 4, 0] = 0.9
    raw[0, :4, 1] = [102, 101, 40, 40]     # same object, lower score -> removed by NMS
    raw[0, 4, 1] = 0.6
    raw[0, :4, 2] = [300, 300, 20, 20]
    raw[0, 5, 2] = 0.05                    # under conf
    boxes, scores, cls = me.decode_yolo(raw, 512, conf=0.1, iou=0.7)
    assert len(scores) == 1 and cls[0] == 0 and boxes[0].tolist() == [80, 80, 120, 120]
    normalized = raw.copy()
    normalized[0, :4] /= 512               # litert-torch: normalized; also transposed layout
    boxes_n, scores_n, _ = me.decode_yolo(normalized.transpose(0, 2, 1), 512, conf=0.1, iou=0.7)
    assert np.allclose(boxes_n, boxes) and np.allclose(scores_n, scores)


def test_letterbox_and_recall_at_thresholds() -> None:
    import numpy as np

    from src.detection import Detection

    canvas, scale, px, py = me.letterbox(np.zeros((360, 640, 3), np.uint8), 512)
    assert canvas.shape == (512, 512, 3) and scale == 0.8 and (px, py) == (0, 112)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    gts = {"a": [("curb", 0.0, 0.8, 1.0, 1.0), ("car", 0.5, 0.5, 0.7, 0.7)]}
    preds = {"a": [Detection("curb", 0.18, 0.0, 0.8, 1.0, 1.0), Detection("car", 0.18, 0.5, 0.5, 0.7, 0.7)]}
    rec = me.recall_at_thresholds(preds, gts, {"curb": 0.15, "car": 0.25}, ["curb", "car", "pole"])
    assert rec == {"curb": 1.0, "car": 0.0, "pole": None}  # curb found at 0.15; car under its 0.25
