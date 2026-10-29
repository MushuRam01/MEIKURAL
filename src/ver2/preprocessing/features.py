"""Estimate Sa once from the complete recording; align every chunk to that Sa."""

import numpy as np

from ...ver1.preprocessing.features import (
    UnusableWindow,
    estimate_tonic,
    make_tonic_estimator,
    normalize,
    spectrogram,
)
from .audio import decoded_blocks, windows


def load_audio(path, config):
    # shortcut: Essentia's standard tonic estimator needs the whole signal; use a streaming equivalent if RAM becomes limiting.
    blocks = list(decoded_blocks(path, config["audio"]))
    if not blocks or not any(len(block) for block in blocks):
        raise ValueError(f"Audio contains no samples: {path}")
    audio = np.concatenate(blocks)
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains nonfinite samples: {path}")
    return audio


def recording_tonic(audio, config, estimator):
    """Use all real samples, including the tail, without artificial window padding."""
    normalized = normalize(audio, config["audio"])
    return estimate_tonic(normalized, config, estimator)


def extract_window(window, config, tonic_hz):
    t = config["tonic"]
    if not np.isfinite(tonic_hz) or not t["minTonicFrequency"] <= tonic_hz <= t["maxTonicFrequency"]:
        raise ValueError("Invalid full-recording tonic")
    real = normalize(window.audio, config["audio"])
    return spectrogram(np.pad(real, (window.left, window.right)), tonic_hz, config), tonic_hz


def iter_features(path, config, estimator=None):
    estimator = make_tonic_estimator(config["tonic"]) if estimator is None else estimator
    audio = load_audio(path, config)
    failure = None
    try:
        tonic_hz = recording_tonic(audio, config, estimator)
    except UnusableWindow as error:
        tonic_hz = None
        failure = f"Full-recording tonic unavailable: {error}"
    rate = config["audio"]["sample_rate"]
    block_size = config["audio"]["decode_block_samples"]
    blocks = (audio[i:i + block_size] for i in range(0, len(audio), block_size))
    for window in windows(blocks, config["audio"]):
        row = {"start_seconds": window.start / rate,
               "valid_start_seconds": max(0, window.start) / rate,
               "valid_seconds": len(window.audio) / rate,
               "left_padding_seconds": window.left / rate,
               "right_padding_seconds": window.right / rate,
               "tonic_hz": tonic_hz, "tonic_scope": "recording", "failure": failure}
        matrix = None
        if failure is None:
            try:
                matrix, _ = extract_window(window, config, tonic_hz)
            except UnusableWindow as error:
                row["failure"] = str(error)
        yield row, matrix
