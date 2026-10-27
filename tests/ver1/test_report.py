"""Folder labels and plots must preserve the existing whole-performance scores."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.ver1.config import load_config
from src.ver1.inference.report import folder_raag, performance_figure, recording_labels


def test_folder_labels_and_discovery(tmp_path):
    classes = load_config()["classes"]
    for folder in ("sree", "miyan_malhar", "bhairav"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "performance.MP3").touch()
        (tmp_path / folder / "notes.txt").touch()
    assert folder_raag("sree", classes) == "Shree"
    assert folder_raag("miyan_malhar", classes) == "Miyan Malhar"
    assert folder_raag("BHAIRAVI", classes) == "Bhairavi"
    assert folder_raag("bhairav", classes) == "bhairav"
    recordings = recording_labels(tmp_path, classes)
    assert len(recordings) == 3
    assert {label for _, label in recordings} == {"Shree", "Miyan Malhar", "bhairav"}
    (tmp_path / "unlabelled.mp3").touch()
    with pytest.raises(ValueError, match="raag folder"):
        recording_labels(tmp_path, classes)


def test_ranking_and_timeline_are_not_recomputed(tmp_path):
    classes = load_config()["classes"]
    ranking = [{"raag": label, "score": score} for label, score in zip(classes, [0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])]
    rows = [{"valid_start_seconds": 0, "valid_seconds": 15, "scores": [0.1] * 7, "failure": None},
            {"valid_start_seconds": 15, "valid_seconds": 30, "scores": None, "failure": "tonic failed"},
            {"valid_start_seconds": 30, "valid_seconds": 30, "scores": [0.2] * 7, "failure": None}]
    result = {"classes": classes, "ranking": ranking, "windows": rows,
              "duration_seconds": 60, "device": "cpu", "usable_windows": 2, "failed_windows": 1}
    fig = performance_figure(result, "sree/a long performance name.mp3", "Shree")
    fig.canvas.draw()
    assert "Correct raag (folder): Shree" in fig._suptitle.get_text()
    assert "Predicted raag: Shree" in fig._suptitle.get_text()
    assert "MATCH" in fig._suptitle.get_text()
    assert len(fig.axes[0].lines) == 7
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[1])
    assert [bar.get_width() for bar in fig.axes[1].patches] == pytest.approx([row["score"] for row in ranking])
    fig.savefig(tmp_path / "report.png")
    assert (tmp_path / "report.png").stat().st_size > 0
    plt.close(fig)


def test_empty_and_failed_performance_figures():
    classes = load_config()["classes"]
    result = {"classes": classes, "ranking": [], "windows": [{"failure": "near-silence"}],
              "duration_seconds": 30, "device": "cpu", "usable_windows": 0, "failed_windows": 1,
              "message": "No prediction"}
    for value, error in ((result, None), (None, "Cannot decode recording")):
        fig = performance_figure(value, "test.mp3", "Shree", error)
        fig.canvas.draw()
        assert "NO PREDICTION" in fig._suptitle.get_text()
        plt.close(fig)


def test_batch_notebook_valid():
    import nbformat

    notebook = nbformat.read("notebooks/04_ver1_handpicked_batch.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs
            compile(cell.source, "batch notebook", "exec")


def test_notebook_batch_continues_after_recording_error(tmp_path):
    import nbformat

    classes = load_config()["classes"]
    root, output = tmp_path / "collection", tmp_path / "reports"
    folder = root / "sree"
    folder.mkdir(parents=True)
    good, bad = folder / "good.mp3", folder / "broken.wav"
    good.touch()
    bad.touch()
    result = {"classes": classes, "ranking": [{"raag": label, "score": 1 / 7} for label in classes],
              "windows": [{"valid_start_seconds": 0, "valid_seconds": 30, "scores": [1 / 7] * 7}],
              "duration_seconds": 30, "device": "cpu", "usable_windows": 1, "failed_windows": 0}

    def predict(path, checkpoint, device):
        if path == bad:
            raise ValueError("Invalid audio")
        return result

    import pickle

    figures = []
    namespace = {"OUTPUT_DIR": output, "TEST_DIR": root, "CHECKPOINT_PATH": "unused.pt",
                 "DEVICE": "cpu", "recordings": [(good, "Shree"), (bad, "Shree")],
                 "predict": predict, "performance_figure": performance_figure,
                 "display": figures.append, "plt": plt, "pickle": pickle}
    notebook = nbformat.read("notebooks/04_ver1_handpicked_batch.ipynb", as_version=4)
    batch = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("OUTPUT_DIR.mkdir"))
    exec(compile(batch, "batch notebook", "exec"), namespace)  # noqa: S102 -- repository-owned cell, stubbed inference.
    assert len(figures) == 2
    assert len(list(output.rglob("*.png"))) == 2
    assert namespace["summaries"][0]["ranking"] == result["ranking"]
    assert "Invalid audio" in namespace["summaries"][1]["error"]
