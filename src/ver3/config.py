"""Version 3: native STFT bins, no pitch estimator or frequency remapping."""

import hashlib
import json
import math
from pathlib import Path

from ..ver1.config import ROOT, resolve_path, write_json

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
__all__ = [
    "DEFAULT_CONFIG",
    "ROOT",
    "feature_fingerprint",
    "feature_shape",
    "load_config",
    "resolve_path",
    "validate_config",
    "write_json",
]


def feature_shape(config):
    a, s = config["audio"], config["spectrogram"]
    samples = round(a["sample_rate"] * a["window_seconds"])
    padding = 2 * (s["n_fft"] // 2) if s["center"] else 0
    return 1 + s["n_fft"] // 2, 1 + (samples + padding - s["n_fft"]) // s["hop_length"]


def validate_config(c):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
        elif isinstance(value, (int, float)):
            require(math.isfinite(value), "Configuration numbers must be finite")

    finite(c)
    require(c.get("preprocessing") == {"version": 3, "representation": "stft_magnitude"},
            "Version 3 requires native STFT features; Version 1/2 checkpoints are incompatible")
    require("tonic" not in c, "Version 3 does not use tonic settings")
    a, s, m, tr = (c[key] for key in ("audio", "spectrogram", "model", "training"))
    require(not {"min_cents", "max_cents_exclusive", "step_cents"} & s.keys(),
            "Version 3 does not use a relative-pitch grid")
    require(len(c["classes"]) == len(set(c["classes"])) == 7, "Version 3 requires seven distinct classes")
    require(all(isinstance(label, str) and label and Path(label).name == label and label not in (".", "..")
                for label in c["classes"]), "Invalid class names")
    for key in ("sample_rate", "decode_block_samples"):
        require(isinstance(a[key], int) and a[key] > 0, f"audio.{key} must be a positive integer")
    require(0 < a["stride_seconds"] <= a["window_seconds"], "Stride must be in (0, window]")
    require(-a["window_seconds"] < a["first_start_seconds"] <= 0, "Invalid first start")
    for key in ("window_seconds", "stride_seconds", "first_start_seconds"):
        require(abs(a[key] * a["sample_rate"] - round(a[key] * a["sample_rate"])) < 1e-6,
                f"audio.{key} must describe a whole number of samples")
    require(round(a["stride_seconds"] * a["sample_rate"]) > 0, "Stride must contain at least one sample")
    require(0 < a["peak_ceiling"] <= 1 and a["silence_rms"] > 0 and a["max_gain_db"] >= 0,
            "Invalid gain limits")
    for key in ("n_fft", "win_length", "hop_length"):
        require(isinstance(s[key], int) and s[key] > 0, f"spectrogram.{key} must be positive")
    require(s["win_length"] <= s["n_fft"] and isinstance(s["center"], bool), "Invalid STFT settings")
    require(s["dynamic_range_db"] > 0 and s["epsilon"] > 0, "Invalid log compression")
    require(len(m["channels"]) == len(m["kernels"]) > 0, "Every CNN block needs channels and a kernel")
    for pair in [m["pool"], m["adaptive_pool"], *m["kernels"]]:
        require(len(pair) == 2 and all(isinstance(v, int) and v > 0 for v in pair), "Invalid CNN dimensions")
    require(all(isinstance(v, int) and v > 0 for v in m["channels"]), "Invalid channels")
    require(all(v % 2 for pair in m["kernels"] for v in pair), "Use odd kernels for same padding")
    height, width = feature_shape(c)
    for _ in m["channels"]:
        height //= m["pool"][0]
        width //= m["pool"][1]
    require(height >= m["adaptive_pool"][0] and width >= m["adaptive_pool"][1], "Input too small for pooling")
    require(0 <= m["dropout"] < 1, "Invalid dropout")
    require(isinstance(m["hidden_features"], int) and m["hidden_features"] > 0,
            "model.hidden_features must be a positive integer")
    require(tr["device"] in ("auto", "cpu", "cuda"), "device must be auto, cpu, or cuda")
    for key in ("epochs", "batch_size"):
        require(isinstance(tr[key], int) and tr[key] > 0, f"training.{key} must be positive")
    require(isinstance(tr["workers"], int) and tr["workers"] >= 0, "Invalid worker count")
    require(tr["learning_rate"] > 0 and tr["weight_decay"] >= 0 and tr["epsilon"] > 0, "Invalid optimizer settings")
    require(len(tr["betas"]) == 2 and all(0 <= b < 1 for b in tr["betas"]), "Invalid Adam betas")
    require(isinstance(c["inference"]["batch_size"], int) and c["inference"]["batch_size"] > 0,
            "Invalid inference batch size")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def feature_fingerprint(config):
    validate_config(config)
    values = {key: config[key] for key in ("preprocessing", "audio", "spectrogram")}
    return hashlib.sha256(json.dumps({"format": 3, **values}, sort_keys=True, allow_nan=False).encode()).hexdigest()
