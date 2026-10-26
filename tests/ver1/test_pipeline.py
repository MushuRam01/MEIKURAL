"""Checks for the signal-processing and model boundaries, without full training."""

import copy
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.ver1.config import (
    feature_fingerprint,
    feature_shape,
    load_config,
    validate_config,
    write_json,
)
from src.ver1.inference.predict import load_model, predict
from src.ver1.preprocessing.audio import decoded_blocks, windows
from src.ver1.preprocessing.features import (
    UnusableWindow,
    extract_window,
    iter_features,
    make_tonic_estimator,
    normalize,
    spectrogram,
)
from src.ver1.preprocessing.prepare import prepare
from src.ver1.training.model import RaagCNN, model_dimensions, select_device
from src.ver1.training.train import FeatureDataset, save_checkpoint, train


@pytest.fixture
def config():
    return load_config()


@pytest.mark.parametrize("duration,starts,padding", [
    (1, [-15], [(15, 14)]),
    (15, [-15], [(15, 0)]),
    (30, [-15, 0], [(15, 0), (0, 0)]),
    (45, [-15, 0, 15], [(15, 0), (0, 0), (0, 0)]),
    (70, [-15, 0, 15, 30, 45], [(15, 0), (0, 0), (0, 0), (0, 0), (0, 5)]),
])
def test_windows(config, duration, starts, padding):
    a = {**config["audio"], "sample_rate": 10}
    audio = np.arange(duration * 10, dtype=np.float32)
    result = list(windows((audio[i:i+17] for i in range(0, len(audio), 17)), a))
    assert [w.start / 10 for w in result] == starts
    assert [(w.left / 10, w.right / 10) for w in result] == padding
    for window in result:
        assert len(window.audio) + window.left + window.right == 300
        np.testing.assert_array_equal(window.audio, audio[max(0, window.start):window.start+300])


def test_normalization(config):
    a = config["audio"]
    result = normalize(np.full(10, 0.02, np.float32), a)
    assert np.sqrt(np.mean(result ** 2)) == pytest.approx(0.1)
    assert normalize(np.full(10, 1e-4, np.float32), a)[0] == pytest.approx(1e-3)
    assert np.max(np.abs(normalize(np.array([10, 0.01], np.float32), a))) <= 0.990001
    with pytest.raises(UnusableWindow, match="silence"):
        normalize(np.zeros(100), a)


def test_relative_pitch_alignment(config):
    rate = config["audio"]["sample_rate"]
    time = np.arange(round(rate * 30)) / rate
    peaks = []
    for tonic in (150, 200):
        audio = np.sin(2 * np.pi * (2 * tonic) * time).astype(np.float32)
        matrix = spectrogram(audio, tonic, config)
        assert matrix.shape == (240, 1292)
        assert np.isfinite(matrix).all() and 0 <= matrix.min() <= matrix.max() <= 1
        peaks.append(int(matrix[:, 20:-20].mean(1).argmax()))
    assert peaks == [96, 96]  # +1200 cents relative to the -1200-cent origin.
    assert not spectrogram(np.zeros_like(audio), 150, config).any()


def test_tonic_estimator_resets_between_windows(config, monkeypatch):
    import essentia.standard

    class StatefulEstimator:
        def __init__(self, **parameters):
            self.at_eof = True

        def reset(self):
            self.at_eof = False

        def __call__(self, audio):
            if self.at_eof:
                raise RuntimeError("No peak locations")
            self.at_eof = True
            return 150

    monkeypatch.setattr(essentia.standard, "TonicIndianArtMusic", StatefulEstimator)
    estimator = make_tonic_estimator(config["tonic"])
    assert estimator(np.ones(100, np.float32)) == 150
    assert estimator(np.ones(100, np.float32)) == 150


def test_invalid_grid_and_fingerprint(config):
    changed = copy.deepcopy(config)
    changed["spectrogram"]["max_cents_exclusive"] = 12000
    with pytest.raises(ValueError, match="Nyquist"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["training"]["epochs"] += 1
    assert feature_fingerprint(config) == feature_fingerprint(changed)
    changed["audio"]["rms_dbfs"] -= 1
    assert feature_fingerprint(config) != feature_fingerprint(changed)


def test_cnn_shapes_and_device(config, monkeypatch):
    rows, total = model_dimensions(config)
    assert total == 37063
    assert [r["shape"] for r in rows if r["layer"].startswith("MaxPool")] == [
        [16, 120, 323], [32, 60, 80], [64, 30, 20]]
    assert rows[-1]["shape"] == [7]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")


def test_optimizer_and_checkpoint(config, tmp_path):
    model = RaagCNN(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.rand(1, 1, *feature_shape(config))
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([1]))
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, config)
    loaded, saved_config, device = load_model(checkpoint, "cpu")
    assert saved_config == config and str(device) == "cpu"
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), loaded(inputs))


def test_decoder_resampling_and_fallback(config, tmp_path):
    rate = 48000
    stereo = np.column_stack((np.full(rate, 0.2), np.full(rate, 0.4))).astype(np.float32)
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, rate, subtype="FLOAT")
    result = np.concatenate(list(decoded_blocks(wav, config["audio"])))
    assert len(result) == 22050
    assert np.mean(result[100:-100]) == pytest.approx(0.3, abs=1e-5)
    if shutil.which("ffmpeg"):
        m4a = tmp_path / "stereo.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), str(m4a)], check=True)
        fallback = np.concatenate(list(decoded_blocks(m4a, config["audio"])))
        assert abs(len(fallback) - 22050) < 1024
        assert np.mean(fallback[100:-100]) == pytest.approx(0.3, abs=0.01)
    corrupt = tmp_path / "broken.wav"
    corrupt.write_text("not audio")
    with pytest.raises(ValueError, match="audio"):
        list(decoded_blocks(corrupt, config["audio"]))


def test_feature_paths_are_identical(config, tmp_path):
    rate = config["audio"]["sample_rate"]
    wave = (0.1 * np.sin(2*np.pi*150*np.arange(rate)/rate)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, wave, rate, subtype="FLOAT")
    estimator = lambda audio: 150.0
    row, matrix = next(iter_features(path, config, estimator))
    window = next(windows(decoded_blocks(path, config["audio"]), config["audio"]))
    expected, tonic = extract_window(window, config, estimator)
    np.testing.assert_array_equal(matrix, expected)
    assert row["tonic_hz"] == tonic and row["right_padding_seconds"] == 14


def test_preparation_training_and_inference(config, tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    rate = config["audio"]["sample_rate"]
    for label in config["classes"]:
        (root / label).mkdir(parents=True)
        sf.write(root / label / "tone.wav", 0.1*np.sin(2*np.pi*150*np.arange(rate)/rate), rate)
    config["paths"] = {"dataset": str(root), "features": str(tmp_path / "features"),
                       "run": str(tmp_path / "run")}
    config["training"].update(epochs=1, batch_size=2, device="cpu")
    monkeypatch.setattr("src.ver1.preprocessing.features.make_tonic_estimator", lambda c: lambda audio: 150)
    monkeypatch.setattr("src.ver1.preprocessing.prepare.make_tonic_estimator", lambda c: lambda audio: 150)
    manifest = prepare(config)
    assert manifest["complete"] and len(manifest["windows"]) == 7
    assert prepare(config) == manifest
    dataset = FeatureDataset(config)
    assert len(dataset) == 7 and dataset[0][0].dtype == torch.float32
    checkpoint = train(config)
    result = predict(root / config["classes"][0] / "tone.wav", checkpoint, "cpu")
    assert result["usable_windows"] == 1 and len(result["ranking"]) == 7
    assert sum(x["score"] for x in result["ranking"]) == pytest.approx(1)
    changed = copy.deepcopy(config)
    changed["audio"]["rms_dbfs"] = -21
    with pytest.raises(ValueError, match="mismatch"):
        prepare(changed)
    with pytest.raises(ValueError, match="mismatch"):
        FeatureDataset(changed)
    with pytest.raises(FileExistsError):
        train(config)
    for seconds in (1, 70):
        path = tmp_path / f"silent{seconds}.wav"
        sf.write(path, np.zeros(rate * seconds), rate)
        result = predict(path, checkpoint, "cpu")
        assert not result["ranking"] and result["duration_seconds"] == seconds
        assert result["failed_windows"] == (1 if seconds == 1 else 5)
    path = tmp_path / "long.wav"
    sf.write(path, 0.1*np.sin(2*np.pi*150*np.arange(rate*70)/rate), rate)
    result = predict(path, checkpoint, "cpu")
    assert result["usable_windows"] == 5 and result["duration_seconds"] == 70
    scores = np.array([row["scores"] for row in result["windows"]])
    weights = [row["valid_seconds"] for row in result["windows"]]
    expected = np.average(scores, axis=0, weights=weights)
    for item in result["ranking"]:
        assert item["score"] == pytest.approx(expected[config["classes"].index(item["raag"])])
    monkeypatch.setattr("src.ver1.preprocessing.features.make_tonic_estimator", lambda c: lambda audio: float("nan"))
    result = predict(path, checkpoint, "cpu")
    assert not result["ranking"] and result["failed_windows"] == 5
    assert all("invalid tonic" in row["failure"] for row in result["windows"])
    manifest["complete"] = False
    write_json(tmp_path / "features/manifest.json", manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        FeatureDataset(config)


def test_notebook_valid():
    import nbformat
    notebook = nbformat.read(Path("notebooks/02_ver1_inference.ipynb"), as_version=4)
    nbformat.validate(notebook)
    assert not any(c.get("outputs") for c in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "inference notebook", "exec")
