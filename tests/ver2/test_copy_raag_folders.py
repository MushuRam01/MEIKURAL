"""Every selected recording is copied, without artist or per-raag limits."""

import json
import sys

import pytest

from src.ver1.preprocessing.copy_raag_folders import (
    RAAG_IDS,
    copy_folders,
    identical_folders,
    plan_copies,
)
from src.ver2.config import load_config
from src.ver2.preprocessing.copy_raag_folders import main


def test_every_recording_dry_run_annotations_and_conflict(tmp_path):
    source, destination = tmp_path / "raw", tmp_path / "v2/set"
    for label in ("Shree", "Jog"):
        for number in range(3):
            folder = source / "same artist" / f"{label} {number}"
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(json.dumps({"raags": [{"uuid": RAAG_IDS[label]}]}))
            (folder / "audio.mp3").write_bytes(b"example audio")
            (folder / "audio.ctonic.txt").write_text("150\n")
    unrelated = source / "unlabelled"
    unrelated.mkdir()
    (unrelated / "metadata.json").write_text(json.dumps({"raags": []}))
    mixed = source / "mixed raags"
    mixed.mkdir()
    (mixed / "metadata.json").write_text(json.dumps({"raags": [{"uuid": RAAG_IDS["Shree"]},
                                                              {"uuid": RAAG_IDS["Jog"]}]}))
    planned = plan_copies(source, destination, ["Shree", "Jog"])
    assert len(planned) == 6
    copy_folders(planned, ["Shree", "Jog"], dry_run=True)
    assert not destination.exists()
    copy_folders(planned, ["Shree", "Jog"])
    assert len(list(destination.rglob("*.mp3"))) == 6
    assert all(identical_folders(left, right) for _, left, right in planned)
    copy_folders(plan_copies(source, destination, ["Shree", "Jog"]), ["Shree", "Jog"])
    assert len(list(destination.rglob("*.mp3"))) == 6
    (planned[0][2] / "audio.mp3").write_bytes(b"different audio")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan_copies(source, destination, ["Shree", "Jog"])


def test_version2_entrypoint_uses_config_not_legacy_raag_list(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "ver2/set"
    config = load_config()
    config["paths"]["dataset"] = str(destination)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    for label in config["classes"]:
        folder = source / "artist" / label
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps({"raags": [{"uuid": RAAG_IDS[label]}]}))
        (folder / "audio.mp3").write_bytes(b"audio")
    command = ["copy_raag_folders", "--config", str(path), "--source", str(source)]
    monkeypatch.setattr(sys, "argv", command + ["--dry-run"])
    main()
    assert not destination.exists()
    monkeypatch.setattr(sys, "argv", command)
    main()
    assert len(list(destination.rglob("*.mp3"))) == 7
