"""Selected passages must not include preceding audio or other inference windows."""

from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from src.ver1.config import load_config
from src.ver1.inference.clip import predict_clip, selected_window
from src.ver1.preprocessing.features import UnusableWindow


def test_selection_boundaries():
    config = load_config()
    config["audio"]["sample_rate"] = 10
    audio = np.arange(700, dtype=np.float32)
    window = selected_window(audio, 20, config)
    np.testing.assert_array_equal(window.audio, audio[200:500])
    assert window.start == 200 and window.left == window.right == 0
    window = selected_window(audio[:100], 0, config)
    assert len(window.audio) == 100 and window.left == 0 and window.right == 200
    window = selected_window(audio, 65, config)
    assert len(window.audio) == 50 and window.right == 250
    for invalid in (-1, float("nan"), float("inf"), 70):
        with pytest.raises(ValueError):
            selected_window(audio, invalid, config)


def test_single_prediction_and_failure(monkeypatch):
    config = load_config()
    rate = config["audio"]["sample_rate"]
    audio = np.ones(rate * 70, dtype=np.float32)
    calls = []

    def extract(window, settings, estimator):
        calls.append(window)
        return np.full((4, 4), 0.123456, dtype=np.float32), 150

    class Model(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.shape == (1, 1, 4, 4)
            assert not self.training
            assert inputs[0, 0, 0, 0].item() == np.float32(np.float16(0.123456))
            return torch.tensor([[0., 0., 0., 0., 5., 0., 0.]])

    monkeypatch.setattr("src.ver1.inference.clip.extract_window", extract)
    result = predict_clip(audio, 20, Model(), config, "cpu", None)
    assert len(calls) == 1 and calls[0].start == 20 * rate
    assert len(calls[0].audio) == 30 * rate
    assert result["start_seconds"] == 20 and result["end_seconds"] == 50
    assert result["ranking"][0]["raag"] == "Todi" and result["tonic_hz"] == 150
    assert sum(r["score"] for r in result["ranking"]) == pytest.approx(1)

    def fail(*args):
        raise UnusableWindow("tonic failed")

    monkeypatch.setattr("src.ver1.inference.clip.extract_window", fail)
    result = predict_clip(audio, 20, Model(), config, "cpu", None)
    assert result["failure"] == "tonic failed" and not result["ranking"]


def test_widget_callbacks(monkeypatch, capsys):
    from src.ver1.inference import interactive

    config = load_config()
    rate = config["audio"]["sample_rate"]
    audio = np.ones(rate * 70, dtype=np.float32) * 0.1
    calls = []
    monkeypatch.setattr(interactive, "load_model", lambda path: (None, config, "cpu"))
    monkeypatch.setattr(interactive, "decoded_blocks", lambda path, settings: iter([audio]))
    monkeypatch.setattr(interactive, "make_tonic_estimator", lambda settings: None)

    def predict(*args):
        calls.append(args[1])
        return {"start_seconds": args[1], "end_seconds": args[1] + 30,
                "device": "cpu", "failure": None, "tonic_hz": 150,
                "right_padding_seconds": 0, "ranking": [{"raag": "Shree", "score": 1.0}]}

    monkeypatch.setattr(interactive, "predict_clip", predict)
    ui = interactive.clip_picker("audio.wav", "model.pt")
    audio_input, _, load, _, controls, label, buttons, _, _ = ui.children
    slider, number = controls.children
    listen, classify = buttons.children
    assert classify.disabled
    load.click()
    assert not classify.disabled and slider.max == pytest.approx(40)
    slider.value = 20
    assert number.value == 20 and not calls
    assert "20.000–50.000" in label.value
    listen.click()
    assert not calls
    classify.click()
    assert calls == [20]
    assert "150.00 Hz" in capsys.readouterr().out
    audio_input.value = "other.wav"
    assert classify.disabled and listen.disabled


def test_interactive_notebook():
    notebook = nbformat.read(Path("notebooks/03_ver1_interactive_clip.ipynb"), as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs
            compile(cell.source, "interactive notebook", "exec")
