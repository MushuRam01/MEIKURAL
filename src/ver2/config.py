"""Version 2 settings: identical CNN, recording-level tonic, separate artifacts."""

import hashlib
import json
from pathlib import Path

from ..ver1.config import ROOT, feature_shape, resolve_path, write_json
from ..ver1.config import validate_config as validate_common

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def validate_config(config):
    if config.get("preprocessing") != {"version": 2, "tonic_scope": "recording"}:
        raise ValueError("Version 2 requires recording-level tonic features; Version 1 checkpoints are incompatible")
    validate_common(config)


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def feature_fingerprint(config):
    validate_config(config)
    values = {key: config[key] for key in ("preprocessing", "audio", "tonic", "spectrogram")}
    payload = json.dumps({"format": 2, **values}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()

__all__ = ['DEFAULT_CONFIG', 'ROOT', 'feature_fingerprint', 'feature_shape', 'load_config', 'resolve_path', 'validate_config', 'write_json']
