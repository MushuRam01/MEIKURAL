"""Train on all cached music: python -m src.ver3.training.train."""

import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import (
    DEFAULT_CONFIG,
    feature_fingerprint,
    feature_shape,
    load_config,
    resolve_path,
    write_json,
)
from .model import RaagCNN, select_device


class FeatureDataset(Dataset):
    def __init__(self, config):
        self.root = resolve_path(config["paths"]["features"])
        manifest = json.loads((self.root / "manifest.json").read_text())
        if not manifest["complete"] or manifest["feature_fingerprint"] != feature_fingerprint(config):
            raise ValueError("Incomplete or mismatched feature cache; run preparation with this config")
        self.rows = [r for r in manifest["windows"] if r["feature"] is not None]
        self.labels = {label: i for i, label in enumerate(config["classes"])}
        self.shape = feature_shape(config)
        counts = np.bincount([self.labels[r["label"]] for r in self.rows], minlength=len(self.labels))
        if np.any(counts == 0):
            raise ValueError("Every class must have at least one usable training window")
        self.weights = torch.tensor(len(self.rows) / (len(self.labels) * counts), dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["feature"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Feature path escapes cache directory")
        matrix = np.load(path, allow_pickle=False).astype(np.float32)
        if matrix.shape != self.shape or not np.isfinite(matrix).all() or matrix.min() < 0 or matrix.max() > 1:
            raise ValueError(f"Invalid cached feature: {path}")
        return torch.from_numpy(matrix[None]), self.labels[row["label"]]


def package_versions():
    return {name: str(importlib.metadata.version(name))
            for name in ("torch", "numpy", "librosa", "soxr", "soundfile")}


def weighted_cross_entropy(logits, labels, weights):
    # Weighted mean reduction would cancel the class weight when batch size is one.
    return F.cross_entropy(logits, labels, weight=weights, reduction="none").mean()


def save_checkpoint(path, model, optimizer, epoch, config):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "classes": config["classes"], "config": config,
                "feature_fingerprint": feature_fingerprint(config),
                "versions": package_versions()}, temporary)
    temporary.replace(path)


def train(config):
    settings = config["training"]
    device = select_device(settings["device"])
    dataset = FeatureDataset(config)
    output = resolve_path(config["paths"]["run"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists():
        raise FileExistsError("Run already has a checkpoint; choose a new paths.run in config.json")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                        num_workers=settings["workers"])
    model = RaagCNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["epsilon"])
    weights = dataset.weights.to(device)
    history = []
    write_json(output / "config.json", config)
    print(f"Training {len(dataset)} windows on {device}; no validation split", flush=True)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = correct = count = 0
        for matrices, labels in loader:
            matrices, labels = matrices.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(matrices)
            loss = weighted_cross_entropy(logits, labels, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; no checkpoint saved for this epoch")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
        row = {"epoch": epoch, "training_loss": total_loss / count,
               "training_accuracy": correct / count}
        history.append(row)
        save_checkpoint(output / "latest.pt", model, optimizer, epoch, config)
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    return output / "latest.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
