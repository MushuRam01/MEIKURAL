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

**2. Check whether GPU training is available.**

```bash
nvidia-smi
python3 -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The current environment was verified with CPU-only Torch. Installing the unified
requirements alone does not replace that build. If `nvidia-smi` works and you
want GPU training, install the CUDA build below, then repeat the Python check.
The CUDA 12.6 wheels require a compatible NVIDIA driver.

```bash
python3 -m pip install --force-reinstall torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

If `nvidia-smi` fails, fix the driver/access issue before expecting GPU training;
installing a Torch wheel does not fix the driver. You may continue on CPU.
With `training.device` set to `"auto"`, training chooses CUDA when available and
otherwise CPU. Feature preparation runs on CPU in either case.

**3. Review `src/ver1/config.json`.**

The defaults use the existing seven-raag collection, 30-second windows with a
15-second stride, batch size 8, and 30 training epochs. Check these fields before
starting:

| Setting | Default | Purpose |
|---|---|---|
| `paths.dataset` | `data/ver1/set` | All selected training recordings |
| `paths.features` | `data/ver1/features` | Generated feature cache |
| `paths.run` | `models/ver1` | Checkpoint and training history |
| `training.device` | `auto` | CUDA when available, otherwise CPU |
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
