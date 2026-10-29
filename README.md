# Meikural

Recognize Hindustani raags from 30-second audio excerpts while learning audio,
signal and time-series processing.

The project currently uses the Saraga Hindustani dataset. External clips supplied
by the project owner will be evaluated separately.

- `docs/MEIKURAL_Project_Proposal.pdf`: original proposal; RaagSense is the earlier project name.
- `notebooks/01_explore_dataset.ipynb`: Saraga exploration.
- `data/raw/saraga_hindustani/`: local dataset, excluded from Git.
- `src/ver1/`: tonic-aligned STFT features, a small CNN, and audio-only inference.
- `docs/ver1.tex`: mathematical and implementation reference, including CNN dimensions.
- `notebooks/02_ver1_inference.ipynb`: infer from an audio file of any positive duration.
- `notebooks/03_ver1_interactive_clip.ipynb`: listen to and classify a chosen 30-second passage.
- `notebooks/04_ver1_handpicked_batch.ipynb`: one combined score timeline/ranking image per handpicked recording.

## Version 2 — full-recording tonic

Version 2 estimates Sa **once from the complete recording**, then uses that
same frequency to align every chunk's STFT. The CNN, gain normalization,
windowing, spectrogram settings, optimizer, and training hyperparameters remain
identical to Version 1. Version 1 code and artifacts are preserved.

**No Version 2 training or full-data feature preparation has been run.** Expand
the datasets first, then run the commands below yourself with `.venv` activated.
All Version 2 settings live in `src/ver2/config.json`:

| Purpose | Default location |
|---|---|
| Training recordings | `data/ver2/set/<configured-raag>/...` |
| External handpicked recordings | `data/test_ver2_handpicked/<raag>/...` |
| Feature cache | `data/ver2/features/` |
| Model and training history | `models/ver2/` |
| Batch reports and confusion matrices | `outputs/ver2/handpicked/` |
| Mathematical reference | `docs/ver2.tex` |

Add performances under the same seven configured raags. You can point
`paths.dataset` at an existing collection instead of copying it, but always keep
Version 2 feature/model outputs separate. Optional Saraga annotations are used
only for tonic diagnostics. There is no internal train/validation/test split.
To compare versions on the existing test collection, set the new batch notebook's
`TEST_DIR` to `ROOT / "data/test_ver1_handpicked"`; its default is the new Version 2
collection. No audio folders have been copied or created automatically.

When the expanded datasets are ready:

```bash
# Preview every matching Saraga recording, without copying:
python3 -m src.ver2.preprocessing.copy_raag_folders --dry-run
# Copy all selected recordings and annotations into data/ver2/set/<raag>/...:
python3 -m src.ver2.preprocessing.copy_raag_folders
# Build features, then train only when you are ready:
python3 -m src.ver2.preprocessing.prepare --diagnose-tonic
python3 -m src.ver2.training.train
```

The copy command reuses Version 1's complete-recording selection and safe copy
logic. It selects **every** single-label recording whose Saraga raag UUID matches
a class in `src/ver2/config.json`, with no per-artist or per-raag limit. The
destination comes from `paths.dataset`. Use `--source /path/to/saraga-collection`
for a different Saraga source. Identical copies are skipped; different existing
contents are never overwritten. The current collection supplies 25 matching
recordings, the same complete selection as Version 1. Unlabelled and multi-raag
recordings are excluded rather than assigned a guessed single label.

Training defaults to required CUDA and 30 epochs, as in Version 1. Version 2
refuses Version 1 caches and checkpoints. Train a new model before using:

```bash
python3 -m jupyterlab notebooks/05_ver2_inference.ipynb
python3 -m jupyterlab notebooks/06_ver2_interactive_clip.ipynb
python3 -m jupyterlab notebooks/07_ver2_handpicked_batch.ipynb
```

The single-audio notebook shows the full-recording Sa and prediction timeline.
The interactive notebook estimates Sa at **Load audio**, then reuses it for every
selected passage. Batch inference includes both song-level and chunk-level
confusion matrices. These notebooks use `models/ver2/latest.pt`; there is no
automatic reuse of Version 1 weights.

The full recording is decoded into RAM for the tonic estimator, so long files
need more memory than Version 1's streaming inference. A single incorrect tonic
can still misalign the whole song; consistency does not guarantee correctness.

Regenerate Version 2's numerical documentation and compile it with:

```bash
python3 -m src.ver2.documentation.generate
mkdir -p outputs/ver2/docs
(cd docs && \
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/ver2/docs ver2.tex && \
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/ver2/docs ver2.tex)
```

Synthetic tests (no training loop) are in `tests/ver2/`:

```bash
python3 -m pytest tests/ver2 -q
```

## Version 1

All 25 selected recordings under `data/ver1/set/` are training data. External
validation and test sets will be supplied later; there is no internal holdout.
Edit **`src/ver1/config.json`** to control paths, preprocessing, tonic estimation,
model dimensions, training, and batch sizes. Relative data/model paths resolve
against the repository root. Raw audio is the only inference input.

### Run in this order

Run the following commands yourself in a terminal. The repository already has
`.venv/`; activate it first so `python3` uses the project environment. Repeat
activation whenever you open a new terminal before running the commands below.

**1. Enter the repository, activate the environment, and install dependencies.**

```bash
cd /home/mushu-cislab/GitHub/MEIKURAL
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` pins project dependencies; pip installs their transitive
dependencies automatically. It includes audio processing, training, notebook,
and development tools. FFmpeg and FFprobe are system programs needed for codecs
not handled by libsndfile. LaTeX is needed only to compile the documentation.

**2. Verify that Torch can use the RTX 4070 before training.**

```bash
nvidia-smi
python3 -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable: do not start training'; print('GPU:', torch.cuda.get_device_name(0))"
```

The requirements explicitly pin `torch==2.9.0+cu126`, so installing them replaces
an existing CPU-only Torch build. The CUDA 12.6 wheels work with this workstation's
NVIDIA driver. A working `nvidia-smi` confirms driver access, but Torch must also
pass the Python check above. To install just the corrected Torch build:

```bash
python3 -m pip install torch==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Training defaults to `training.device: "cuda"` and fails immediately if CUDA is
unavailable; it cannot silently switch to CPU. If you deliberately want CPU
training, set this to `"cpu"`. `"auto"` remains available for automatic CPU fallback.
Feature preparation and loading cached arrays still use CPU; the CNN and its
training batches run on CUDA. Driver access must also be available inside any
container or sandbox used to launch training.

**3. Review `src/ver1/config.json`.**

The defaults use the existing seven-raag collection, 30-second windows with a
15-second stride, batch size 8, and 30 training epochs. Check these fields before
starting:

| Setting | Default | Purpose |
|---|---|---|
| `paths.dataset` | `data/ver1/set` | All selected training recordings |
| `paths.features` | `data/ver1/features` | Generated feature cache |
| `paths.run` | `models/ver1` | Checkpoint and training history |
| `training.device` | `cuda` | Require CUDA; fail instead of silently using CPU |
| `training.batch_size` | `8` | Reduce if GPU memory is insufficient |
| `training.epochs` | `30` | Fixed training duration; no internal validation |

The selected dataset is already copied, so you do not need to rerun the download,
exploration notebook, or recording-copy script.

**4. Check the environment and implementation.**

```bash
python3 -m pip check
python3 -m pytest tests/ver1 -q
python3 -m ruff check src/ver1 tests/ver1
```

Resolve failures before preparing the full collection.

**5. Prepare training features and inspect tonic diagnostics.**

```bash
python3 -m src.ver1.preprocessing.prepare --diagnose-tonic
```

This is the first full-data processing command. It reads raw audio, generates
windows, estimates tonic from audio, computes tonic-aligned STFT matrices, and
caches them as float16 NumPy files under `data/ver1/features/`. The manifest
records source recordings, timestamps, tonics, padding, and skipped windows.
The command can take time and several GB of disk space; wait for completion.

Review skipped-window reasons and the printed annotation comparisons before
training. Diagnostics use supplied annotations only for comparison, never to
correct features. The small implementation smoke sample had substantial tonic
errors (only 4/14 estimates within 50 cents), so usable features alone do not
establish correct tonic alignment. See `docs/ver1_verification.md`.

A completed matching cache is reused on reruns. Changing audio, tonic, or STFT
settings, or changing the source collection, requires a new `paths.features`.

**6. Train the CNN.**

```bash
python3 -m src.ver1.training.train
```

This uses all usable training windows, prints the selected device, and saves
`models/ver1/latest.pt`, `history.json`, and a configuration snapshot. Wait for
all configured epochs to finish. Loss and accuracy are **training metrics**;
external validation and testing have not been supplied yet.

The startup line must say `on cuda`. If you previously killed a CPU run, keep
the prepared feature cache; it is independent of training device. If that run
saved `latest.pt`, change `paths.run` to a fresh directory before starting again.

An existing checkpoint is protected against overwrite. Choose a new `paths.run`
for a new experiment; automatic resume is not implemented.

**7. Run inference on your own audio.**

Register the project environment once, then launch the inference notebook:

```bash
python3 -m ipykernel install --user --name meikural --display-name "Python (MEIKURAL)"
python3 -m jupyterlab notebooks/02_ver1_inference.ipynb
```

Select the **Python (MEIKURAL)** kernel. Edit `AUDIO_PATH` to your audio file and
`CHECKPOINT_PATH` to the checkpoint from step 6, then run the cells in order.
Any positive audio duration is supported through windowing and padding.

The inference notebook displays per-window predictions, failures, estimated tonics,
and a duration-weighted overall ranking. Scores rank only the seven supported
raags; they are not calibrated confidence or unknown-raag detection. The checkpoint
supplies the model and preprocessing settings even if the editable JSON changes.

Alternatively, replace the example audio path and use the terminal:

```bash
python3 -m src.ver1.inference.predict /absolute/path/to/your_audio.wav models/ver1/latest.pt
```

### Choose a 30-second passage interactively

With `.venv` activated, install the updated requirements and open the clip picker:

```bash
python3 -m pip install -r requirements.txt
python3 -m jupyterlab notebooks/03_ver1_interactive_clip.ipynb
```

Select the **Python (MEIKURAL)** kernel and run the cells in order. Enter audio
and checkpoint paths, click **Load audio**, choose a start time using the slider
or exact-seconds field, and click **Play selected clip** to listen. Click
**Predict selected clip** to classify exactly that passage. Moving the slider
clears old results but never starts inference automatically.

This predicts one selected window, without the whole-song notebook's initial
half-padded window or averaging other passages. Audio shorter than 30 seconds is
right-padded. The checkpoint controls duration and preprocessing (30 seconds for
the Version 1 model). The song is decoded once into RAM for responsive seeking.
Restart JupyterLab after installing widgets if controls do not appear.

Vocal passages may help isolate the musical material you want to inspect, but
the model still receives the complete audio spectrum, including accompaniment.
Selecting vocals does not separate the singer or correct a wrong tonic estimate.

### Report every handpicked performance

With `.venv` activated, launch:

```bash
python3 -m jupyterlab notebooks/04_ver1_handpicked_batch.ipynb
```

Select **Python (MEIKURAL)** and run the cells in order. The defaults scan all
supported audio files recursively under `data/test_ver1_handpicked/`, using the
first folder below it as the correct raag label. Review the file list before
running the batch cell. `sree` maps to `Shree`, and `miyan_malhar` to `Miyan Malhar`;
unknown folders remain distinct, including `bhairav` versus `Bhairavi`.

For each recording, one Matplotlib image contains the filename, correct raag,
predicted raag, match status, seven-score timeline, and full duration-weighted
overall ranking. The shared inference function supplies the same weighted mean
as notebook 02. Each image is displayed inline and saved under
`outputs/ver1/handpicked/<raag-folder>/<audio-filename>.png`. No recordings are
pooled together. Failed recordings receive an error image without stopping the
remaining batch. Rerunning overwrites the corresponding report PNGs, not audio.

The final two cells display and save `confusion_matrix_songs.png` and
`confusion_matrix_chunks.png` in the same output directory. The first counts one
final duration-weighted prediction per song; the second counts each successfully
predicted window once against its song's folder label, including padded edges.
Rows are ground truth and columns are predictions. Failed predictions are
excluded with counts shown; overlapping windows are correlated. Restart the
kernel and rerun the updated batch once to populate the new chunk-label summaries.

### Share the trained model and infer on a laptop

The trained `models/ver1/latest.pt` checkpoint is approximately 446 KB, so it can
be committed to ordinary Git without Git LFS. The ignore rules allow this file,
its `config.json`, and `history.json`; other model runs, datasets, and feature
caches stay ignored. The checkpoint includes weights, class order, and the saved
preprocessing configuration. You do not need the Saraga dataset to run inference.

Include these files and ignore-rule changes in your next commit, then push the
commit before cloning or pulling on the laptop:

```bash
git add .gitignore .gitattributes models/ver1/latest.pt models/ver1/config.json models/ver1/history.json
```

For a **Linux laptop without an NVIDIA GPU**, use Python 3.12 and Bash. From the
cloned repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
# Use the same dependency file, excluding only the workstation Torch/index lines.
python3 -m pip install -r <(sed '/^torch==/d; /^--extra-index-url /d' requirements.txt)
python3 -m src.ver1.inference.predict /absolute/path/to/your_audio.mp3 models/ver1/latest.pt --device cpu
```

The process-substitution command above requires Bash. It keeps one requirements
file while avoiding installation of the workstation CUDA build on a CPU laptop.
FFmpeg and FFprobe must be installed separately for fallback audio decoding.
No feature preparation or training is needed. Saved CUDA weights can load on CPU;
the saved training device does not force inference onto CUDA.

For the interactive notebook, register the laptop environment and launch:

```bash
python3 -m ipykernel install --user --name meikural --display-name "Python (MEIKURAL)"
python3 -m jupyterlab notebooks/03_ver1_interactive_clip.ipynb
```

Select **Python (MEIKURAL)**, set the laptop's audio path, and keep the checkpoint
path at `models/ver1/latest.pt`. Notebook inference selects CPU automatically when
CUDA is unavailable. Laptop audio stays local; only explicitly committed files
are transferred through Git. CPU inference will generally be slower than GPU.

### Optional: regenerate mathematical documentation

After changing configuration, regenerate the numerical tables and compile twice
to update the contents and references. The subshell keeps your terminal in the
repository root:

```bash
python3 -m src.ver1.documentation.generate
mkdir -p outputs/ver1/docs
(cd docs && \
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/ver1/docs ver1.tex && \
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/ver1/docs ver1.tex)
```

The PDF is written to `outputs/ver1/docs/ver1.pdf`.

See `docs/ver1_verification.md` for implementation verification and limitations.
