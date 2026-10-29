"""Classify a selected window using the already estimated full-recording tonic."""

import numpy as np
import torch

from ...ver1.inference.clip import selected_window
from ..preprocessing.features import UnusableWindow, extract_window


def predict_clip(audio, start_seconds, model, config, device, tonic_hz):
    window = selected_window(audio, start_seconds, config)
    rate = config["audio"]["sample_rate"]
    result = {"start_seconds": window.start / rate,
              "end_seconds": (window.start + len(window.audio)) / rate,
              "valid_seconds": len(window.audio) / rate,
              "right_padding_seconds": window.right / rate,
              "device": str(device), "tonic_hz": tonic_hz, "tonic_scope": "recording",
              "ranking": [], "failure": None}
    try:
        matrix, result["tonic_hz"] = extract_window(window, config, tonic_hz)
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
