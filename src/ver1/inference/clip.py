"""Classify exactly one user-selected window, without whole-song aggregation."""

import numpy as np
import torch

from ..preprocessing.audio import Window
from ..preprocessing.features import UnusableWindow, extract_window


def selected_window(audio, start_seconds, config):
    rate = config["audio"]["sample_rate"]
    if audio.ndim != 1 or not len(audio):
        raise ValueError("Expected nonempty mono audio at the checkpoint sample rate")
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Start time must be finite and nonnegative")
    start = round(start_seconds * rate)
    if start >= len(audio):
        raise ValueError("Start time must be before the end of the audio")
    length = round(config["audio"]["window_seconds"] * rate)
    real = np.asarray(audio[start:start + length], dtype=np.float32)
    if not np.isfinite(real).all():
        raise ValueError("Selected audio contains nonfinite samples")
    return Window(start, real, 0, length - len(real))


def predict_clip(audio, start_seconds, model, config, device, estimator):
    window = selected_window(audio, start_seconds, config)
    rate = config["audio"]["sample_rate"]
    result = {"start_seconds": window.start / rate,
              "end_seconds": (window.start + len(window.audio)) / rate,
              "valid_seconds": len(window.audio) / rate,
              "right_padding_seconds": window.right / rate,
              "device": str(device), "tonic_hz": None, "ranking": [], "failure": None}
    try:
        matrix, result["tonic_hz"] = extract_window(window, config, estimator)
    except UnusableWindow as error:
        result["failure"] = str(error)
        return result
    matrix = matrix.astype(np.float16).astype(np.float32)
    inputs = torch.from_numpy(matrix[None, None]).to(device)
    model.eval()
    with torch.inference_mode():
        scores = model(inputs).softmax(1)[0].cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("Model produced nonfinite scores")
    result["ranking"] = sorted(
        [{"raag": label, "score": float(score)}
         for label, score in zip(config["classes"], scores, strict=True)],
        key=lambda row: row["score"], reverse=True)
    return result
