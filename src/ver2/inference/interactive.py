"""Select passages while holding the full-song tonic fixed."""

import pickle

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import Audio, clear_output, display

from ..config import resolve_path
from ..preprocessing.features import load_audio, make_tonic_estimator, recording_tonic
from .clip import predict_clip, selected_window
from .predict import load_model


def clip_picker(audio_path, checkpoint_path):
    audio_input = widgets.Text(value=str(audio_path), description="Audio path:", layout=widgets.Layout(width="100%"))
    checkpoint_input = widgets.Text(value=str(checkpoint_path), description="Checkpoint:", layout=widgets.Layout(width="100%"))
    load_button = widgets.Button(description="Load audio", button_style="info")
    slider = widgets.FloatSlider(description="Start (s):", min=0, max=0, step=0.1,
                                 continuous_update=False, readout_format=".1f",
                                 layout=widgets.Layout(width="75%"), disabled=True)
    number = widgets.BoundedFloatText(description="Exact start:", min=0, max=0, disabled=True)
    widgets.link((slider, "value"), (number, "value"))
    listen_button = widgets.Button(description="Play selected clip", disabled=True)
    predict_button = widgets.Button(description="Predict selected clip", button_style="success", disabled=True)
    interval = widgets.Label(value="Load an audio file and checkpoint to begin.")
    status, preview, predictions = widgets.Output(), widgets.Output(), widgets.Output()
    state = {}

    def selection_changed(change=None):
        preview.clear_output()
        predictions.clear_output()
        if not state:
            return
        window = selected_window(state["audio"], slider.value, state["config"])
        rate = state["config"]["audio"]["sample_rate"]
        start = window.start / rate
        end = start + len(window.audio) / rate
        interval.value = f"Selected: {start:.3f}–{end:.3f} s | Right padding: {window.right / rate:.3f} s"

    def invalidate(change=None):
        state.clear()
        for control in (slider, number, listen_button, predict_button):
            control.disabled = True
        preview.clear_output()
        predictions.clear_output()
        status.clear_output()
        interval.value = "Paths changed. Click Load audio."

    def load_clicked(button):
        invalidate()
        load_button.disabled = True
        with status:
            clear_output(wait=True)
            print("Loading audio and checkpoint…")
            try:
                model, config, device = load_model(resolve_path(checkpoint_input.value))
                audio = load_audio(resolve_path(audio_input.value), config)
                estimator = make_tonic_estimator(config["tonic"])
                print("Estimating Sa from the full recording…")
                tonic_hz = recording_tonic(audio, config, estimator)
                state.update(audio=audio, config=config, model=model, device=device, tonic_hz=tonic_hz)
                slider.value = 0
                number.value = 0
                maximum = max(0, (len(audio) - round(config["audio"]["window_seconds"] * config["audio"]["sample_rate"]))
                              / config["audio"]["sample_rate"])
                # Set both bounds before enabling the linked controls.
                number.max = maximum
                slider.max = maximum
                for control in (slider, number, listen_button, predict_button):
                    control.disabled = False
                clear_output(wait=True)
                print(f"Duration: {len(audio) / config['audio']['sample_rate']:.3f} s | Device: {device}")
                print(f"Sa from the full recording: {tonic_hz:.2f} Hz (fixed for every selection)")
                print(f"Select a {config['audio']['window_seconds']:g}-second passage, listen, then predict.")
                selection_changed()
            except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
                state.clear()
                print(f"{type(error).__name__}: {error}")
            finally:
                load_button.disabled = False

    def listen_clicked(button):
        if not state:
            return
        with preview:
            clear_output(wait=True)
            window = selected_window(state["audio"], slider.value, state["config"])
            # Playback is the selected real audio; feature extraction adds padding.
            playback = window.audio / max(1.0, float(np.max(np.abs(window.audio))))
            display(Audio(playback, rate=state["config"]["audio"]["sample_rate"], normalize=False))

    def predict_clicked(button):
        if not state:
            return
        predict_button.disabled = True
        with predictions:
            clear_output(wait=True)
            print("Classifying the selected passage using the full-recording Sa…")
            try:
                result = predict_clip(state["audio"], slider.value, state["model"],
                                      state["config"], state["device"], state["tonic_hz"])
                clear_output(wait=True)
                print(f"Clip: {result['start_seconds']:.3f}–{result['end_seconds']:.3f} s | Device: {result['device']}")
                if result["failure"]:
                    print(f"No prediction: {result['failure']}")
                else:
                    print(f"Full-recording Sa: {result['tonic_hz']:.2f} Hz | Right padding: {result['right_padding_seconds']:.3f} s")
                    display(pd.DataFrame(result["ranking"]))
                    print("Comparative scores across seven raags; not calibrated confidence.")
            except (ValueError, RuntimeError) as error:
                print(f"{type(error).__name__}: {error}")
            finally:
                predict_button.disabled = False

    audio_input.observe(invalidate, names="value")
    checkpoint_input.observe(invalidate, names="value")
    slider.observe(selection_changed, names="value")
    load_button.on_click(load_clicked)
    listen_button.on_click(listen_clicked)
    predict_button.on_click(predict_clicked)
    return widgets.VBox([audio_input, checkpoint_input, load_button, status,
                         widgets.HBox([slider, number]), interval,
                         widgets.HBox([listen_button, predict_button]), preview, predictions])
