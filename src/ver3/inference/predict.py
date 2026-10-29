"""Version 3 inference using native STFT magnitudes without pitch alignment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import feature_fingerprint, validate_config
from ..preprocessing.features import iter_features
from ..training.model import RaagCNN, select_device


def load_model(checkpoint, device="auto"):
    selected = select_device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    validate_config(config)
    if state["classes"] != config["classes"] or state["feature_fingerprint"] != feature_fingerprint(config):
        raise ValueError("Checkpoint configuration or class mapping is inconsistent")
    model = RaagCNN(config)
    model.load_state_dict(state["model"])
    return model.to(selected).eval(), config, selected


def predict(audio_path, checkpoint_path, device="auto"):
    """Return per-window scores/failures and a duration-weighted overall ranking."""
    model, config, selected = load_model(checkpoint_path, device)
    rows, pending = [], []
    weighted = np.zeros(len(config["classes"]), dtype=np.float64)
    total_weight = 0.0

    def flush():
        nonlocal total_weight
        if not pending:
            return
        batch = torch.from_numpy(np.stack([matrix for _, matrix in pending])[:, None]).to(selected)
        with torch.inference_mode():
            scores = model(batch).softmax(1).cpu().numpy()
        for (row, _), score in zip(pending, scores, strict=True):
            if not np.isfinite(score).all():
                raise RuntimeError("Model produced nonfinite scores")
            row["scores"] = score.tolist()
            row["top_raag"] = config["classes"][int(score.argmax())]
            weighted[:] += score * row["valid_seconds"]
            total_weight += row["valid_seconds"]
        pending.clear()

    for row, matrix in iter_features(audio_path, config):
        row.update(scores=None, top_raag=None)
        rows.append(row)
        if matrix is not None:
            # Match the training cache's float16 quantization exactly.
            pending.append((row, matrix.astype(np.float16).astype(np.float32)))
            if len(pending) >= config["inference"]["batch_size"]:
                flush()
    flush()
    overall = weighted / total_weight if total_weight else None
    ranking = [] if overall is None else sorted(
        [{"raag": label, "score": float(score)}
         for label, score in zip(config["classes"], overall, strict=True)],
        key=lambda item: item["score"], reverse=True)
    duration = max(row["valid_start_seconds"] + row["valid_seconds"] for row in rows)
    usable = sum(row["scores"] is not None for row in rows)
    return {"audio": str(Path(audio_path).expanduser().resolve()), "duration_seconds": duration,
            "representation": "stft_magnitude",
            "device": str(selected), "classes": config["classes"], "usable_windows": usable,
            "failed_windows": len(rows) - usable, "windows": rows, "ranking": ranking,
            "message": "Comparative seven-class scores, not calibrated probabilities."
            if ranking else "No prediction: every window was near-silent."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(predict(args.audio, args.checkpoint, args.device), indent=2))


if __name__ == "__main__":
    main()
