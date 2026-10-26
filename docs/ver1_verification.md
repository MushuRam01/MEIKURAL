# Version 1 implementation verification

Verified on 27 October 2026 with Python 3.12, Torch 2.9.0+cpu,
Essentia 2.1b6.dev1389, and the existing project audio packages.

## Checks

- `python -m pytest tests/ver1 -q`: **15 passed**.
- `ruff check src/ver1 tests/ver1`: passed.
- `python -m pip check`: no broken requirements.
- Default numerical feature shape: **240 × 1,292**; CNN: **37,063 parameters**.
- Documentation tables generated from the JSON and actual Torch modules.
- LaTeX compiled with no warnings or overflowing boxes; the nine-page PDF and
  its model/settings tables were visually inspected. Build output is ignored at
  `outputs/ver1/docs/ver1.pdf`.

The tests cover exact window boundaries, padding-independent normalization,
gain limits, silence, synthetic relative-pitch alignment, configuration/cache
mismatches, model shapes, one optimization step and checkpoint round trip,
CPU fallback, native and FFmpeg decoding, resampling/channel averaging, shared
feature extraction, short/long inference, failures, duration-weighted aggregation,
and notebook validity. The tonic reset regression test verifies that the stateful
Essentia algorithm is reset before each independent window.

## Real-audio smoke run

Used one source recording per class, taking seconds 60–90. Excerpts and the
single-epoch model were kept in temporary directories, separate from production
feature and model paths. Every excerpt was decoded from the actual local Saraga
collection; no supplied tonic was used to construct its features.

- Seven 30-second excerpts produced **14 usable windows, zero skipped**.
- Actual Essentia estimation, STFT, remapping, float16 cache, and training ran.
- One CPU epoch, batch size 2, completed and saved its checkpoint/history.
- Weighted training loss: **1.97094**; training accuracy: **1/14**. These numbers
  check finite computation only and are not evidence of recognition accuracy.
- Reloaded the checkpoint and predicted from raw Shree excerpt audio:
  two successful windows with seven finite scores per window.
- The inference notebook was executed against the temporary checkpoint; its
  executed copy is under ignored `outputs/ver1/smoke/`. The delivered notebook
  remains clean, unexecuted, and without user-specific audio paths.

An initial smoke run exposed state retained by Essentia's frame cutter. Resetting
the estimator before each clip resolved the resulting false no-peak failures.

## Tonic diagnostic findings

Compared the 14 independently estimated window tonics against their recording's
provided tonic annotation **after** extraction:

| Diagnostic | Observed result |
|---|---:|
| Median absolute raw cents error | 500.51 cents |
| Median absolute octave-corrected cents error | 10.72 cents |
| Nearest octave displacement nonzero | 5 / 14 windows |
| Raw error within 50 cents | 4 / 14 windows |

These overlapping excerpts are a tiny, correlated diagnostic sample. The broad
65–500 Hz search returned octave errors and some non-octave errors; successful
extraction does **not** establish that the tonic is correct. Octave correction
above is a diagnostic calculation only. Features are not corrected using
annotations. Assess tonic accuracy on a broader sample before interpreting full
training results; the requested default estimator and search interval remain
unchanged.

## Not run

No full-dataset feature preparation, full training, internal holdout evaluation,
external validation, or external testing was performed. GPU execution was not
verified: `nvidia-smi` could not communicate with the driver in this session.
CPU execution and automatic fallback were verified. External dataset ingestion,
unknown-raag rejection, and score calibration are outside this implementation.

### GPU setup follow-up

An unsandboxed `nvidia-smi` check confirmed the RTX 4070 (12 GB) and NVIDIA driver
580.173.02 are working. The project environment still contained Torch 2.9.0+cpu,
which explains the CPU training run despite working hardware. Requirements now
pin `torch==2.9.0+cu126`; training defaults to explicit `cuda` so it fails rather
than silently falling back. CPU and `auto` remain selectable. The CUDA package
installation and actual GPU training are user-run; no training was restarted.
