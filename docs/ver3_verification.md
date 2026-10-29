# Version 3 implementation verification

Version 3 removes tonic finding and frequency remapping and uses a larger CNN.
Initial implementation checks did not run real-data preparation or a training
loop. A subsequent user-launched training run is recorded below.

## Checks

- **Nine synthetic checks pass** in `tests/ver3/`; Version 3 source/tests pass Ruff.
- Gain normalization is the exact shared implementation from Version 1.
  Sample rate, RMS target, maximum gain, peak ceiling, silence threshold, window
  duration/stride, start position, and padding behavior retain their earlier values.
- Every native STFT bin is retained. Feature values match direct Librosa STFT
  magnitude plus the documented intensity compression, without axis interpolation.
- A guard test makes earlier tonic estimators and aligned-spectrogram functions
  fail if called; Version 3 still produces all expected synthetic windows.
- The default input is **4,097 × 1,292**. A meta-tensor forward pass exercises
  every layer at that full input size without allocating large activation tensors.
- CNN filters are **32/64/128**, the hidden classifier is **256 units**, and the
  parameter count is **521,095**. Actual post-pool shapes match the generated table.
- Batch-one class-weight tests confirm weights do not cancel out of the loss.
- Temporary synthetic feature preparation, cache reuse/mismatch rejection, dummy
  untrained checkpoint reload, and rejection of older checkpoint versions pass.
- Whole-audio and selected-clip inference wiring use the same no-tonic features.
  Interactive controls invoke prediction only after the explicit button click.
- All three Version 3 notebooks validate, compile, have clean outputs, and use
  Version 3 modules. Batch input defaults to the existing shared handpicked
  collection and includes both confusion-matrix cells.
- Mathematical tables are generated from `src/ver3/config.json` and actual model
  modules; `docs/ver3.tex` compiles without warnings or overflowing boxes.
  The rendered document and model table were visually inspected.

## Completed user training and expanded test collection

- The pipeline log and saved history confirm **30 completed CUDA epochs** on
  **3,471 windows**, with batch size **2** and zero skipped training windows.
- Final training loss: **0.0998414**; training accuracy: **95.9954%**. These are
  training metrics, not external evaluation results.
- The trained `models/ver3/latest.pt` checkpoint loads on CPU and includes Marwa
  among its seven supported labels.
- The shared test loader discovers **20 recordings**: Bhairavi 2, Jog 2, Lalit 2,
  Marwa 3, Miyan Malhar 5, Shree 3, and Todi 3.
- All three new Marwa files pass a decoder spot check. One file named `.mp3`
  actually contains Opus audio; the existing FFmpeg fallback decodes it. Files
  were not renamed, transcoded, or added to training.
- No full handpicked inference batch or external accuracy evaluation was run
  during this collection check. Rerun the notebook's discovery and inference cells
  to regenerate per-recording reports and both confusion matrices.

Small synthetic tests use a reduced FFT size to keep checks inexpensive; default
geometry is tested separately with meta tensors.

## Costs and limits

The native float16 cache is approximately **10.59 MB per window**, or **36.75 GB**
at the previous collection's 3,471 windows. Actual size depends on the dataset.
Current training batch size is 2; inference remains at 1.
The first convolution output alone is about 677.5 MB in float32, excluding other
activations, gradients, and backend workspaces. Synthetic passes reserved 2.77 GiB
at batch 1 and 6.42 GiB at batch 2; the full run completed on the RTX 4070 at batch 2.
Larger batch sizes remain untested.

Absolute pitch, accompaniment, and timbre remain in the input. A larger model can
overfit limited independent recordings; no accuracy improvement is claimed. All
earlier model versions and datasets are preserved.
