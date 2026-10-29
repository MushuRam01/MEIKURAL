"""Cache absolute-frequency STFT training windows: python -m src.ver3.preprocessing.prepare."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import (
    DEFAULT_CONFIG,
    feature_fingerprint,
    load_config,
    resolve_path,
    write_json,
)
from .features import iter_features

FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma"}


def recording_list(config):
    root = resolve_path(config["paths"]["dataset"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training dataset not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in FORMATS and p.is_file())
    records = []
    for path in files:
        relative = path.relative_to(root)
        label = relative.parts[0]
        if label not in config["classes"]:
            raise ValueError(f"Audio is not under a configured raag folder: {path}")
        stat = path.stat()
        # shortcut: size/mtime detect source edits; use content hashes for immutable audits.
        records.append({"recording_id": relative.as_posix(), "label": label,
                        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    missing = set(config["classes"]) - {r["label"] for r in records}
    if missing:
        raise ValueError(f"No recordings for: {', '.join(sorted(missing))}")
    return root, records


def prepare(config):
    root, records = recording_list(config)
    destination = resolve_path(config["paths"]["features"]).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Feature cache and source dataset must be separate, non-nested directories")
    fingerprint = feature_fingerprint(config)
    manifest_path = destination / "manifest.json"
    manifest = {"feature_fingerprint": fingerprint, "config": config,
                "dataset": str(root), "recordings": records,
                "complete": False, "windows": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["feature_fingerprint"] != fingerprint or previous["recordings"] != records or previous["dataset"] != str(root):
            raise ValueError("Cache/config or source mismatch; choose a new features directory in config.json")
        if previous["complete"]:
            for row in previous["windows"]:
                if row["feature"] and not (destination / row["feature"]).is_file():
                    raise ValueError("Cache is incomplete; choose a new features directory")
            print(f"Reusing complete feature cache: {destination}")
            return previous
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Feature directory contains files without a manifest; choose a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    for index, record in enumerate(records):
        print(f"Extracting absolute-frequency STFT windows: {record['recording_id']}", flush=True)
        for number, (row, matrix) in enumerate(iter_features(root / record["recording_id"], config)):
            row.update(recording_id=record["recording_id"], label=record["label"], feature=None)
            if matrix is not None:
                name = f"{index:04d}_{number:06d}.npy"
                temporary = destination / (name + ".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, matrix.astype(np.float16), allow_pickle=False)
                temporary.replace(destination / name)
                row["feature"] = name
            else:
                print(f"SKIP {record['recording_id']} @ {row['start_seconds']:g}s: {row['failure']}")
            manifest["windows"].append(row)
        write_json(manifest_path, manifest)
        print(f"Prepared {index + 1}/{len(records)}: {record['recording_id']}", flush=True)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    usable = sum(row["feature"] is not None for row in manifest["windows"])
    print(f"Training cache: {usable} usable, {len(manifest['windows']) - usable} skipped")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
