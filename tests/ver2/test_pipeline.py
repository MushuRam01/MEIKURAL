"""Version 2 contracts, using synthetic audio; never run a training loop."""

import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest
import soundfile as sf
import torch

from src.ver1.config import feature_fingerprint as v1_fingerprint
from src.ver1.config import load_config as v1_config
from src.ver1.inference.predict import load_model as v1_load_model
from src.ver1.training.model import RaagCNN as V1CNN
from src.ver2.config import (
    feature_fingerprint,
    feature_shape,
    load_config,
    validate_config,
)
from src.ver2.inference.clip import predict_clip
from src.ver2.inference.predict import load_model, predict
from src.ver2.preprocessing.features import iter_features, recording_tonic
from src.ver2.preprocessing.prepare import prepare
from src.ver2.training.model import RaagCNN, model_dimensions
from src.ver2.training.train import FeatureDataset, save_checkpoint


@pytest.fixture
def config():
    return load_config()


def test_only_tonic_scope_and_artifact_paths_change(config):
    old = v1_config()
    for section in ("classes", "audio", "tonic", "spectrogram", "model", "training", "inference"):
        assert config[section] == old[section]
    assert RaagCNN is V1CNN
    assert feature_shape(config) == (240, 1292)
    assert model_dimensions(config)[1] == 37063
    assert feature_fingerprint(config) != v1_fingerprint(old)
    assert config["paths"] == {"dataset": "data/ver2/set", "features": "data/ver2/features", "run": "models/ver2"}
    with pytest.raises(ValueError, match="incompatible"):
        validate_config(old)


def tone_file(path, seconds, rate=22050):
    audio = (0.03 * np.sin(2 * np.pi * 150 * np.arange(round(rate * seconds)) / rate)).astype(np.float32)
    sf.write(path, audio, rate, subtype="FLOAT")
    return audio


def test_full_recording_estimated_once_and_shared_by_all_windows(config, tmp_path):
    path = tmp_path / "long.wav"
    tone_file(path, 70)
    calls = []

    def estimate(audio):
        calls.append(len(audio))
        return 150

    rows = list(iter_features(path, config, estimate))
    assert calls == [70 * config["tonic"]["sampleRate"]]
    assert len(rows) == 5
    assert [r["start_seconds"] for r, _ in rows] == [-15, 0, 15, 30, 45]
    assert all(r["tonic_hz"] == 150 and r["tonic_scope"] == "recording" for r, _ in rows)
    assert all(matrix.shape == feature_shape(config) for _, matrix in rows)
    assert rows[-1][0]["right_padding_seconds"] == 5


def test_tonic_failure_has_no_per_window_fallback(config, tmp_path):
    path = tmp_path / "long.wav"
    tone_file(path, 70)
    calls = []

    def fail(audio):
        calls.append(len(audio))
        raise RuntimeError("No peak locations")

    rows = list(iter_features(path, config, fail))
    assert len(calls) == 1 and len(rows) == 5
    assert all(matrix is None and "Full-recording tonic unavailable" in row["failure"] for row, matrix in rows)
    sf.write(path, np.zeros(22050), 22050)
    assert next(iter_features(path, config, fail))[1] is None
    assert len(calls) == 1  # Global near-silence is rejected before calling Essentia.


def test_preparation_cache_is_separate_and_records_one_tonic(config, tmp_path, monkeypatch):
    root = tmp_path / "set"
    for label in config["classes"]:
        (root / label).mkdir(parents=True)
        tone_file(root / label / "song.wav", 1)
    config["paths"].update(dataset=str(root), features=str(tmp_path / "features"))
    calls = []

    def estimate(audio):
        calls.append(len(audio))
        return 150

    monkeypatch.setattr("src.ver2.preprocessing.prepare.make_tonic_estimator", lambda c: estimate)
    manifest = prepare(config)
    assert len(calls) == 7
    assert len(manifest["recording_tonics"]) == 7
    assert set(manifest["recording_tonics"].values()) == {150}
    assert len(FeatureDataset(config)) == 7
    assert prepare(config) == manifest and len(calls) == 7
    changed = copy.deepcopy(config)
    changed["tonic"]["minTonicFrequency"] = 70
    with pytest.raises(ValueError, match="mismatch"):
        prepare(changed)
    manifest["feature_fingerprint"] = v1_fingerprint(config)
    (tmp_path / "features/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        FeatureDataset(config)


def test_checkpoint_version_boundary_without_training(config, tmp_path):
    model = RaagCNN(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, model, optimizer, 0, config)
    restored, saved, device = load_model(path, "cpu")
    assert saved == config and str(device) == "cpu"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match="inconsistent"):
        v1_load_model(path, "cpu")
    old = v1_config()
    state = torch.load(path, weights_only=True)
    state.update(config=old, feature_fingerprint=v1_fingerprint(old))
    torch.save(state, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(path, "cpu")


class ConstantCNN(torch.nn.Module):
    def forward(self, matrix):
        return torch.zeros((len(matrix), 7), device=matrix.device)


def test_single_inference_and_interactive_clip_share_recording_tonic(config, tmp_path, monkeypatch):
    path = tmp_path / "song.wav"
    audio = tone_file(path, 45)
    calls = []

    def estimate(signal):
        calls.append(len(signal))
        return 150

    monkeypatch.setattr("src.ver2.preprocessing.features.make_tonic_estimator", lambda c: estimate)
    monkeypatch.setattr("src.ver2.inference.predict.load_model", lambda p, d: (ConstantCNN(), config, "cpu"))
    result = predict(path, "unused.pt", "cpu")
    assert len(calls) == 1 and result["recording_tonic_hz"] == 150
    assert result["usable_windows"] == 3
    assert all(row["tonic_hz"] == 150 for row in result["windows"])
    tonic = recording_tonic(audio, config, estimate)
    before = len(calls)
    for start in (0, 15):
        selected = predict_clip(audio, start, ConstantCNN(), config, "cpu", tonic)
        assert selected["tonic_hz"] == 150 and selected["tonic_scope"] == "recording"
        assert selected["end_seconds"] == start + 30
    assert len(calls) == before  # Changing the selected passage never estimates Sa.


def test_widget_load_estimates_once_not_on_slider_changes(config, monkeypatch):
    from src.ver2.inference import interactive

    audio = np.full(22050 * 70, 0.1, np.float32)
    calls = []
    selections = []
    monkeypatch.setattr(interactive, "load_model", lambda p: (None, config, "cpu"))
    monkeypatch.setattr(interactive, "load_audio", lambda p, c: audio)
    monkeypatch.setattr(interactive, "make_tonic_estimator", lambda c: None)

    def tonic(signal, settings, estimator):
        calls.append(len(signal))
        return 150

    def selected(signal, start, model, settings, device, tonic_hz):
        selections.append((start, tonic_hz))
        return {"start_seconds": start, "end_seconds": start + 30, "device": "cpu",
                "tonic_hz": tonic_hz, "right_padding_seconds": 0, "failure": None,
                "ranking": [{"raag": "Shree", "score": 1.0}]}

    monkeypatch.setattr(interactive, "recording_tonic", tonic)
    monkeypatch.setattr(interactive, "predict_clip", selected)
    ui = interactive.clip_picker("song.wav", "v2.pt")
    ui.children[2].click()
    slider = ui.children[4].children[0]
    button = ui.children[6].children[1]
    for start in (10, 20):
        slider.value = start
        button.click()
    assert calls == [len(audio)] and selections == [(10, 150), (20, 150)]


def test_version2_notebooks_are_clean_and_use_version2_pipeline():
    paths = ["05_ver2_inference.ipynb", "06_ver2_interactive_clip.ipynb", "07_ver2_handpicked_batch.ipynb"]
    for name in paths:
        notebook = nbformat.read(Path("notebooks") / name, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "src.ver2." in source and "src.ver1." not in source
        assert "models/ver2/latest.pt" in source
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert not cell.outputs
                compile(cell.source, name, "exec")
    batch = nbformat.read("notebooks/07_ver2_handpicked_batch.ipynb", as_version=4)
    assert "Song-level confusion matrix" in batch.cells[-2].source
    assert "Chunk-level confusion matrix" in batch.cells[-1].source
