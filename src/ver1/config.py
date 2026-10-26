"""One configuration shared by preparation, training, inference, and documentation."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def feature_shape(config):
    a, s = config["audio"], config["spectrogram"]
    rows = math.ceil((s["max_cents_exclusive"] - s["min_cents"]) / s["step_cents"])
    samples = round(a["sample_rate"] * a["window_seconds"])
    padded = samples + (2 * (s["n_fft"] // 2) if s["center"] else 0)
    return rows, 1 + (padded - s["n_fft"]) // s["hop_length"]


def validate_config(c):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                finite_tree(item)
        elif isinstance(value, (int, float)):
            require(math.isfinite(value), "Configuration numbers must be finite")

    finite_tree(c)
    a, s, t, m, tr = (c[k] for k in ("audio", "spectrogram", "tonic", "model", "training"))
    require(len(c["classes"]) == 7 and len(set(c["classes"])) == 7,
            "Version 1 requires seven distinct class names")
    require(all(isinstance(x, str) and x and Path(x).name == x
                and x not in (".", "..") for x in c["classes"]), "Invalid class names")
    for key in ("sample_rate", "decode_block_samples"):
        require(isinstance(a[key], int) and a[key] > 0, f"audio.{key} must be a positive integer")
    require(0 < a["stride_seconds"] <= a["window_seconds"], "Stride must be in (0, window]")
    require(-a["window_seconds"] < a["first_start_seconds"] <= 0, "Invalid first start")
    for key in ("window_seconds", "stride_seconds", "first_start_seconds"):
        require(abs(a[key] * a["sample_rate"] - round(a[key] * a["sample_rate"])) < 1e-6,
                f"audio.{key} must describe a whole number of samples")
    require(round(a["stride_seconds"] * a["sample_rate"]) > 0,
            "Stride must contain at least one sample")
    require(0 < a["peak_ceiling"] <= 1 and a["silence_rms"] > 0 and a["max_gain_db"] >= 0,
            "Invalid audio gain limits")
    for key in ("n_fft", "win_length", "hop_length"):
        require(isinstance(s[key], int) and s[key] > 0, f"spectrogram.{key} must be positive")
    require(s["win_length"] <= s["n_fft"], "win_length must not exceed n_fft")
    require(isinstance(s["center"], bool), "center must be boolean")
    require(s["step_cents"] > 0 and s["max_cents_exclusive"] > s["min_cents"], "Invalid pitch grid")
    require(s["dynamic_range_db"] > 0 and s["epsilon"] > 0, "Invalid log compression")
    require(0 < t["minTonicFrequency"] < t["maxTonicFrequency"], "Invalid tonic range")
    rows, frames = feature_shape(c)
    highest = s["min_cents"] + (rows - 1) * s["step_cents"]
    require(t["maxTonicFrequency"] * 2 ** (highest / 1200) < a["sample_rate"] / 2,
            "Pitch grid exceeds Nyquist at the maximum tonic")
    require(len(m["channels"]) == len(m["kernels"]) > 0, "Each CNN block needs a channel count and kernel")
    for pair in [m["pool"], m["adaptive_pool"], *m["kernels"]]:
        require(len(pair) == 2 and all(isinstance(v, int) and v > 0 for v in pair),
                "CNN dimensions must be positive integer pairs")
    require(all(isinstance(v, int) and v > 0 for v in m["channels"]), "Invalid channel counts")
    require(all(v % 2 for pair in m["kernels"] for v in pair), "Use odd kernels for same padding")
    for _ in m["channels"]:
        rows //= m["pool"][0]
        frames //= m["pool"][1]
    require(rows >= m["adaptive_pool"][0] and frames >= m["adaptive_pool"][1],
            "Input is too small for configured pooling")
    require(0 <= m["dropout"] < 1, "Invalid dropout")
    require(tr["device"] in ("auto", "cpu", "cuda"), "device must be auto, cpu, or cuda")
    for key in ("epochs", "batch_size"):
        require(isinstance(tr[key], int) and tr[key] > 0, f"training.{key} must be positive")
    require(isinstance(tr["workers"], int) and tr["workers"] >= 0, "Invalid worker count")
    require(tr["learning_rate"] > 0 and tr["weight_decay"] >= 0 and tr["epsilon"] > 0,
            "Invalid optimizer settings")
    require(len(tr["betas"]) == 2 and all(0 <= b < 1 for b in tr["betas"]), "Invalid Adam betas")
    require(isinstance(c["inference"]["batch_size"], int) and c["inference"]["batch_size"] > 0,
            "Invalid inference batch size")


def feature_fingerprint(config):
    # Bump the format when feature mathematics or decoder semantics change.
    values = {k: config[k] for k in ("audio", "tonic", "spectrogram")}
    payload = json.dumps({"format": 1, **values}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path, value):
    """Publish complete JSON, including manifests and epoch histories."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
