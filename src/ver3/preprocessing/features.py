"""Gain-normalized, zero-padded audio -> native linear-frequency STFT magnitude."""

import librosa
import numpy as np

from ...ver1.preprocessing.features import UnusableWindow, normalize
from .audio import decoded_blocks, windows


def spectrogram(audio, config):
    s = config["spectrogram"]
    magnitude = np.abs(librosa.stft(audio, n_fft=s["n_fft"], win_length=s["win_length"],
                                   hop_length=s["hop_length"], window=s["window"],
                                   center=s["center"], pad_mode="constant"))
    peak = float(magnitude.max())
    if peak <= s["epsilon"]:
        return np.zeros_like(magnitude, dtype=np.float32)
    # Compress intensities in place; never interpolate, shift, or discard frequency bins.
    np.maximum(magnitude, s["epsilon"], out=magnitude)
    magnitude /= peak
    np.log10(magnitude, out=magnitude)
    magnitude *= 20
    np.clip(magnitude, -s["dynamic_range_db"], 0, out=magnitude)
    magnitude += s["dynamic_range_db"]
    magnitude /= s["dynamic_range_db"]
    return magnitude.astype(np.float32, copy=False)


def extract_window(window, config):
    real = normalize(window.audio, config["audio"])
    return spectrogram(np.pad(real, (window.left, window.right)), config)


def iter_features(path, config):
    rate = config["audio"]["sample_rate"]
    for window in windows(decoded_blocks(path, config["audio"]), config["audio"]):
        row = {"start_seconds": window.start / rate,
               "valid_start_seconds": max(0, window.start) / rate,
               "valid_seconds": len(window.audio) / rate,
               "left_padding_seconds": window.left / rate,
               "right_padding_seconds": window.right / rate,
               "representation": "stft_magnitude", "failure": None}
        try:
            matrix = extract_window(window, config)
        except UnusableWindow as error:
            matrix = None
            row["failure"] = str(error)
        yield row, matrix


def load_audio(path, config):
    # shortcut: interactive seeking caches one song in RAM; use streaming inference for very long audio.
    blocks = list(decoded_blocks(path, config["audio"]))
    if not blocks or not any(len(block) for block in blocks):
        raise ValueError(f"Audio contains no samples: {path}")
    audio = np.concatenate(blocks)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains nonfinite samples")
    return audio
