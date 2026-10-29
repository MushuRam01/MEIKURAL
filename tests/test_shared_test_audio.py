"""All inference workflows share raw test audio and keep versioned artifacts."""

import ast
import importlib
from pathlib import Path

import nbformat
import pytest

from src.ver1.config import load_config


@pytest.mark.parametrize("version,numbers", [(1, (2, 3, 4)), (2, (5, 6, 7)), (3, (8, 9, 10))])
def test_notebook_audio_roots_and_checkpoint_paths(version, numbers):
    root = Path("/example/MEIKURAL")
    paths = {"run": f"models/ver{version}"}
    for number in numbers:
        file = next(Path("notebooks").glob(f"{number:02d}_*.ipynb"))
        notebook = nbformat.read(file, as_version=4)
        nbformat.validate(notebook)
        namespace = {"ROOT": root, "config": {"paths": paths}, "resolve_path": lambda value: root / value}
        audio_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            for statement in tree.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if not isinstance(target, ast.Name) or target.id not in {"TEST_DIR", "AUDIO_PATH", "CHECKPOINT_PATH", "OUTPUT_DIR"}:
                    continue
                # Normalized paths later in a cell rely on Path being available too.
                namespace["Path"] = Path
                module = ast.Module(body=[statement], type_ignores=[])
                exec(compile(module, str(file), "exec"), namespace)  # noqa: S102 -- only repository-owned path assignments.
                if target.id == "TEST_DIR":
                    assert namespace["TEST_DIR"] == root / "data/test_ver1_handpicked"
                elif target.id == "AUDIO_PATH":
                    audio_count += 1
                    assert namespace["AUDIO_PATH"].is_relative_to(root / "data/test_ver1_handpicked")
                elif target.id == "CHECKPOINT_PATH":
                    assert namespace["CHECKPOINT_PATH"] == root / f"models/ver{version}/latest.pt"
                elif target.id == "OUTPUT_DIR":
                    assert namespace["OUTPUT_DIR"] == root / f"outputs/ver{version}/handpicked"
        assert audio_count > 0 or "OUTPUT_DIR" in namespace


@pytest.mark.parametrize("version", [1, 2, 3])
def test_every_version_discovers_new_marwa_performances(version, tmp_path):
    folder = tmp_path / "marwa" / "nested"
    folder.mkdir(parents=True)
    for name in ("artist_one.mp3", "artist_two.MP3", "artist_three.wav"):
        (folder / name).touch()
    module = importlib.import_module(f"src.ver{version}.inference.report")
    rows = module.recording_labels(tmp_path, load_config()["classes"])
    assert len(rows) == 3 and all(label == "Marwa" for _, label in rows)
