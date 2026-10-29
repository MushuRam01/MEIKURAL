"""Native-STFT contracts and larger CNN shape checks; no training loop."""

import copy
import json
from pathlib import Path

import librosa
import nbformat
import numpy as np
import pytest
import soundfile as sf
import torch

from src.ver1.preprocessing.features import normalize as v1_normalize
from src.ver2.config import load_config as v2_config
from src.ver3.config import (
    feature_fingerprint,
    feature_shape,
    load_config,
    validate_config,
)
from src.ver3.inference.clip import predict_clip
from src.ver3.inference.predict import load_model, predict
from src.ver3.preprocessing.features import iter_features, normalize, spectrogram
from src.ver3.preprocessing.prepare import prepare
from src.ver3.training.model import RaagCNN, model_dimensions
from src.ver3.training.train import (
    FeatureDataset,
    save_checkpoint,
    weighted_cross_entropy,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_config(config):
    config["spectrogram"].update(n_fft=512, win_length=512)
    return config


def tone_file(path, seconds, rate=22050):
    audio = (0.03 * np.sin(2 * np.pi * 440 * np.arange(round(rate * seconds)) / rate)).astype(np.float32)
    sf.write(path, audio, rate, subtype="FLOAT")
    return audio


def test_volume_gain_padding_and_window_settings_are_preserved(config):
    old = v2_config()
    assert config["audio"] == old["audio"]
    assert normalize is v1_normalize
    for key in ("n_fft", "win_length", "hop_length", "window", "center", "dynamic_range_db", "epsilon"):
        assert config["spectrogram"][key] == old["spectrogram"][key]
    for key in ("epochs", "device", "learning_rate", "weight_decay", "betas", "epsilon", "seed", "workers"):
        assert config["training"][key] == old["training"][key]
    assert "tonic" not in config
    assert feature_shape(config) == (4097, 1292)
    for invalid in (old, {**config, "tonic": {}}):
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_native_stft_matches_direct_computation(small_config):
    rate = small_config["audio"]["sample_rate"]
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(rate * 30) / rate)).astype(np.float32)
    actual = spectrogram(audio, small_config)
    s = small_config["spectrogram"]
    magnitude = np.abs(librosa.stft(audio, n_fft=s["n_fft"], win_length=s["win_length"],
                                   hop_length=s["hop_length"], window=s["window"], center=s["center"],
                                   pad_mode="constant"))
    expected = (np.clip(20 * np.log10(np.maximum(magnitude, s["epsilon"]) / magnitude.max()),
                        -s["dynamic_range_db"], 0) + s["dynamic_range_db"]) / s["dynamic_range_db"]
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert actual.shape == feature_shape(small_config)
    assert np.isfinite(actual).all() and 0 <= actual.min() <= actual.max() <= 1
    assert not spectrogram(np.zeros_like(audio), small_config).any()


def test_default_full_input_and_larger_classifier_dimensions(config):
    rows, total = model_dimensions(config)
    assert total == 521095
    assert [r["shape"] for r in rows if r["layer"].startswith("MaxPool")] == [
        [32, 2048, 323], [64, 1024, 80], [128, 512, 20]]
    # Meta tensors exercise every layer at the real input size without allocating GPU/CPU activations.
    with torch.device("meta"):
        model = RaagCNN(config)
        shapes = []
        for layer in model.features:
            if isinstance(layer, torch.nn.MaxPool2d):
                layer.register_forward_hook(lambda module, inputs, output: shapes.append(list(output.shape[1:])))
        output = model(torch.zeros(1, 1, *feature_shape(config)))
        assert output.shape == (1, 7)
        assert shapes == [[32, 2048, 323], [64, 1024, 80], [128, 512, 20]]
        assert model.head[2].in_features == 1536 and model.head[2].out_features == 256


def test_class_weights_remain_effective_for_one_window_batches():
    logits = torch.zeros(1, 7)
    labels = torch.tensor([1])
    weights = torch.ones(7)
    baseline = weighted_cross_entropy(logits, labels, weights)
    weights[1] = 3
    assert weighted_cross_entropy(logits, labels, weights).item() == pytest.approx(3 * baseline.item())


def test_streamed_windows_have_no_tonic_calls(small_config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Version 3 must not estimate or align tonic")

    monkeypatch.setattr("src.ver1.preprocessing.features.make_tonic_estimator", forbidden)
    monkeypatch.setattr("src.ver1.preprocessing.features.estimate_tonic", forbidden)
    monkeypatch.setattr("src.ver1.preprocessing.features.spectrogram", forbidden)
    path = tmp_path / "song.wav"
    tone_file(path, 70)
    rows = list(iter_features(path, small_config))
    assert [row["start_seconds"] for row, _ in rows] == [-15, 0, 15, 30, 45]
    assert rows[0][0]["left_padding_seconds"] == 15
    assert rows[-1][0]["right_padding_seconds"] == 5
    assert all(matrix.shape == feature_shape(small_config) and "tonic_hz" not in row for row, matrix in rows)
    sf.write(path, np.zeros(22050), 22050)
    row, matrix = next(iter_features(path, small_config))
    assert matrix is None and "near-silence" in row["failure"]


def test_cache_and_checkpoint_isolation_without_training(small_config, tmp_path):
    config = small_config
    root = tmp_path / "set"
    for label in config["classes"]:
        (root / label).mkdir(parents=True)
        tone_file(root / label / "song.wav", 1)
    config["paths"].update(dataset=str(root), features=str(tmp_path / "features"))
    manifest = prepare(config)
    assert manifest["complete"] and len(manifest["windows"]) == 7
    assert "recording_tonics" not in manifest
    assert len(FeatureDataset(config)) == 7
    assert prepare(config) == manifest
    changed = copy.deepcopy(config)
    changed["spectrogram"]["dynamic_range_db"] = 60
    with pytest.raises(ValueError, match="mismatch"):
        prepare(changed)
    model = RaagCNN(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, model, optimizer, 0, config)
    restored, saved, device = load_model(path, "cpu")
    assert saved == config and str(device) == "cpu"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])
    torch.save({"config": v2_config()}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(path, "cpu")
    manifest["feature_fingerprint"] = "old-version-fingerprint"
    (tmp_path / "features/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        FeatureDataset(config)
    assert feature_fingerprint(config) != feature_fingerprint(changed)


class ConstantCNN(torch.nn.Module):
    def forward(self, matrix):
        return torch.zeros(len(matrix), 7, device=matrix.device)


def test_whole_audio_and_selected_clip_use_same_no_tonic_features(small_config, tmp_path, monkeypatch):
    path = tmp_path / "song.wav"
    audio = tone_file(path, 45)
    monkeypatch.setattr("src.ver3.inference.predict.load_model", lambda p, d: (ConstantCNN(), small_config, "cpu"))
    result = predict(path, "unused.pt", "cpu")
    assert result["usable_windows"] == 3 and "recording_tonic_hz" not in result
    assert sum(row["score"] for row in result["ranking"]) == pytest.approx(1)
    selected = predict_clip(audio, 15, ConstantCNN(), small_config, "cpu")
    assert selected["start_seconds"] == 15 and selected["end_seconds"] == 45
    assert "tonic_hz" not in selected and len(selected["ranking"]) == 7


def test_widget_predicts_without_full_song_analysis(small_config, monkeypatch):
    from src.ver3.inference import interactive

    audio = np.full(22050 * 70, 0.1, np.float32)
    calls = []
    monkeypatch.setattr(interactive, "load_model", lambda p: (None, small_config, "cpu"))
    monkeypatch.setattr(interactive, "load_audio", lambda p, c: audio)

    def selected(signal, start, model, config, device):
        calls.append(start)
        return {"start_seconds": start, "end_seconds": start + 30, "device": "cpu", "failure": None,
                "right_padding_seconds": 0, "ranking": [{"raag": "Shree", "score": 1.0}]}

    monkeypatch.setattr(interactive, "predict_clip", selected)
    ui = interactive.clip_picker("audio.wav", "v3.pt")
    ui.children[2].click()
    ui.children[4].children[0].value = 20
    assert not calls
    ui.children[6].children[1].click()
    assert calls == [20]


def test_version3_notebooks():
    names = ["08_ver3_inference.ipynb", "09_ver3_interactive_clip.ipynb", "10_ver3_handpicked_batch.ipynb"]
    for name in names:
        notebook = nbformat.read(Path("notebooks") / name, as_version=4)
        nbformat.validate(notebook)
        assert "models/ver3/latest.pt" in "\n".join(c.source for c in notebook.cells)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert not cell.outputs
                assert "tonic_hz" not in cell.source and "src.ver2." not in cell.source
                compile(cell.source, name, "exec")
    batch = nbformat.read("notebooks/10_ver3_handpicked_batch.ipynb", as_version=4)
    assert "data/test_ver1_handpicked" in batch.cells[3].source
    assert "Song-level confusion matrix" in batch.cells[-2].source
    assert "Chunk-level confusion matrix" in batch.cells[-1].source
