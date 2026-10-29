"""Discover folder-labelled recordings and plot one report per performance."""

import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ..preprocessing.prepare import FORMATS


def confusion_figure(truth, predictions, classes, title):
    """Rows are folder labels; columns are predictions, with raw sample counts."""
    if len(truth) != len(predictions):
        raise ValueError("Each ground-truth label must have one prediction")
    labels = list(classes) + sorted((set(truth) | set(predictions)) - set(classes))
    counts = (confusion_matrix(truth, predictions, labels=labels) if truth
              else np.zeros((len(labels), len(labels)), dtype=int))
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    ConfusionMatrixDisplay(counts, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False,
        im_kw={"vmin": 0, "vmax": max(1, int(counts.max()))})
    ax.set(title=title, xlabel="Predicted raag", ylabel="Correct raag (folder)")
    return fig


def folder_raag(name, classes):
    normalized = " ".join(name.replace("_", " ").replace("-", " ").split()).casefold()
    if normalized == "sree":
        normalized = "shree"
    return next((label for label in classes if label.casefold() == normalized), name)


def recording_labels(root, classes):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Test collection not found: {root}")
    recordings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FORMATS:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(f"Place recordings inside a raag folder: {path}")
        recordings.append((path, folder_raag(relative.parts[0], classes)))
    if not recordings:
        raise ValueError(f"No supported audio files found under {root}")
    return recordings


def performance_figure(result, audio_name, correct_raag, error=None):
    """Use predict()'s ranking directly; never reweight or pool across recordings."""
    ranking = result["ranking"] if result is not None else []
    predicted = ranking[0]["raag"] if ranking else "No prediction"
    if not ranking:
        verdict = "NO PREDICTION"
    elif correct_raag not in result["classes"]:
        verdict = "FOLDER LABEL OUTSIDE MODEL CLASSES"
    else:
        verdict = "MATCH" if predicted == correct_raag else "MISMATCH"
    fig, (timeline, overall) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.4, 1]}, layout="constrained")
    filename = textwrap.fill(str(audio_name), width=100)
    fig.suptitle(f"{filename}\nCorrect raag (folder): {correct_raag} | Predicted raag: {predicted}\n{verdict}",
                 fontsize=12, fontweight="bold")
    timeline.set(xlabel="Window midpoint in recording (seconds)", ylabel="Window score",
                 ylim=(0, 1), title="Scores across the performance")
    timeline.grid(alpha=0.2)
    overall.set_title("Overall ranking\n(real-duration-weighted window scores)", fontsize=10)
    if result is None:
        message = textwrap.fill(f"Inference failed: {error}", width=75)
        timeline.text(0.5, 0.5, message, ha="center", va="center", transform=timeline.transAxes)
        overall.axis("off")
        fig.supxlabel("Recording failed; the batch continues with the next file.", fontsize=9)
        return fig

    palette = plt.get_cmap("tab10")
    colors = {label: palette(index) for index, label in enumerate(result["classes"])}
    if ranking:
        times = [row["valid_start_seconds"] + row["valid_seconds"] / 2 for row in result["windows"]]
        # NaNs leave visible gaps where feature extraction failed.
        scores = np.array([row["scores"] if row["scores"] is not None
                           else [np.nan] * len(result["classes"]) for row in result["windows"]])
        for index, label in enumerate(result["classes"]):
            timeline.plot(times, scores[:, index], marker=".", markersize=3,
                          linewidth=1.3, label=label, color=colors[label])
        timeline.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9)
        bars = overall.barh([row["raag"] for row in ranking], [row["score"] for row in ranking],
                            color=[colors[row["raag"]] for row in ranking])
        overall.invert_yaxis()
        overall.set(xlim=(0, 1.15), xlabel="Weighted mean score")
        overall.set_xticks([0, 0.25, 0.5, 0.75, 1])
        overall.bar_label(bars, labels=[f"{row['score']:.6f}" for row in ranking], padding=4, fontsize=9)
    else:
        timeline.text(0.5, 0.5, "No usable windows", ha="center", va="center", transform=timeline.transAxes)
        failures = Counter(row["failure"] for row in result["windows"] if row["failure"])
        message = "\n".join(textwrap.fill(f"{count} window(s): {reason}", 45)
                            for reason, count in failures.items())
        overall.text(0.5, 0.5, message or result["message"], ha="center", va="center",
                     transform=overall.transAxes, fontsize=9)
        overall.axis("off")
    fig.supxlabel(
        f"Duration: {result['duration_seconds']:.2f} s | Device: {result['device']} | "
        f"Usable windows: {result['usable_windows']} | Failed windows: {result['failed_windows']}\n"
        "Scores compare seven supported raags; they are not calibrated probabilities. Folder labels are supplied ground truth.",
        fontsize=9)
    return fig
