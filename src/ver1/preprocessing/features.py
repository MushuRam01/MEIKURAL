"""The same raw-audio -> tonic-relative STFT matrix for training and inference."""

import librosa
import numpy as np
import soxr

from .audio import decoded_blocks, windows


class UnusableWindow(ValueError):
    """A silent passage or failed tonic estimate, recorded rather than guessed."""


def normalize(real, config):
    rms = float(np.sqrt(np.mean(np.square(real, dtype=np.float64))))
    if rms < config["silence_rms"]:
        raise UnusableWindow(f"near-silence (RMS {rms:.3g})")
    gain = min(10 ** (config["rms_dbfs"] / 20) / rms,
               10 ** (config["max_gain_db"] / 20),
               config["peak_ceiling"] / float(np.max(np.abs(real))))
    return (real * gain).astype(np.float32)


def make_tonic_estimator(config):
    try:
        from essentia.standard import TonicIndianArtMusic
    except ImportError as error:
        raise RuntimeError("Install essentia==2.1b6.dev1389 to estimate tonic from audio") from error
    # Construct once per preparation/prediction run; invalid parameters fail early.
    algorithm = TonicIndianArtMusic(**config)

    def estimate(audio):
        # Essentia's frame cutter retains its EOF position between compute calls.
        algorithm.reset()
        return algorithm(audio)

    return estimate


def estimate_tonic(real, config, estimator):
    a, t = config["audio"], config["tonic"]
    audio = soxr.resample(real, a["sample_rate"], t["sampleRate"], quality="HQ")
    try:
        tonic = float(estimator(np.asarray(audio, dtype=np.float32)))
    except RuntimeError as error:
        raise UnusableWindow(f"tonic estimation failed: {error}") from error
    if not np.isfinite(tonic) or not t["minTonicFrequency"] <= tonic <= t["maxTonicFrequency"]:
        raise UnusableWindow(f"invalid tonic estimate: {tonic}")
    return tonic


def spectrogram(audio, tonic, config):
    a, s = config["audio"], config["spectrogram"]
    magnitude = np.abs(librosa.stft(audio, n_fft=s["n_fft"], win_length=s["win_length"],
                                   hop_length=s["hop_length"], window=s["window"],
                                   center=s["center"], pad_mode="constant"))
    cents = np.arange(s["min_cents"], s["max_cents_exclusive"], s["step_cents"])
    # Interpolate along frequency only; preserve every STFT time frame.
    positions = tonic * 2 ** (cents / 1200) * s["n_fft"] / a["sample_rate"]
    if not np.isfinite(positions).all() or positions.min() < 0 or positions.max() >= magnitude.shape[0] - 1:
        raise ValueError("Tonic-relative grid falls outside the STFT frequency range")
    low = positions.astype(int)
    fraction = (positions - low)[:, None]
    relative = magnitude[low] * (1 - fraction) + magnitude[low + 1] * fraction
    peak = float(relative.max())
    if peak <= s["epsilon"]:
        return np.zeros(relative.shape, np.float32)
    db = 20 * np.log10(np.maximum(relative, s["epsilon"]) / peak)
    return ((np.clip(db, -s["dynamic_range_db"], 0) + s["dynamic_range_db"])
            / s["dynamic_range_db"]).astype(np.float32)


def extract_window(window, config, estimator):
    real = normalize(window.audio, config["audio"])
    tonic = estimate_tonic(real, config, estimator)
    padded = np.pad(real, (window.left, window.right))
    return spectrogram(padded, tonic, config), tonic


def iter_features(path, config, estimator=None):
    estimator = make_tonic_estimator(config["tonic"]) if estimator is None else estimator
    rate = config["audio"]["sample_rate"]
    for window in windows(decoded_blocks(path, config["audio"]), config["audio"]):
        row = {"start_seconds": window.start / rate,
               "valid_start_seconds": max(0, window.start) / rate,
               "valid_seconds": len(window.audio) / rate,
               "left_padding_seconds": window.left / rate,
               "right_padding_seconds": window.right / rate}
        try:
            matrix, row["tonic_hz"] = extract_window(window, config, estimator)
            row["failure"] = None
        except UnusableWindow as error:
            matrix = None
            row.update(tonic_hz=None, failure=str(error))
        yield row, matrix
