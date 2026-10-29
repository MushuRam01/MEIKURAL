# Version 2 implementation verification

Version 2 has been implemented without preparing the real datasets or training
a CNN. No production Version 2 checkpoint or feature cache has been generated.

## Checks

- Ten focused tests pass using synthetic audio, temporary caches, dummy model
  outputs, and untrained checkpoint round trips.
- CNN class identity, all model/training/audio/STFT defaults, and the seven class
  labels match Version 1. The default input remains **240 × 1,292** and the model
  remains **37,063 parameters**.
- A synthetic 70-second recording reaches the tonic estimator as one complete
  signal. Exactly one estimator call supplies the same Sa to all five windows,
  including the padded first and last windows.
- A recording-wide estimation failure marks every window unusable without
  falling back to independent window estimates. Silence is rejected before the
  estimator is called.
- Cache tests retain one tonic per recording and reject mismatched artifacts.
- Version 1 and Version 2 checkpoint loaders reject the other version's feature
  fingerprint/configuration. No weights are copied between versions.
- Selected-clip and widget tests reuse the full-song Sa after slider changes,
  without further tonic-estimator calls.
- All three Version 2 notebooks validate, have clean outputs, and compile. The
  batch notebook includes both song-level and chunk-level confusion matrices.
- Version 2 source and tests pass Ruff. The documentation settings/model tables
  are generated from the unified Version 2 configuration and real CNN modules.
- `docs/ver2.tex` compiles without warnings or overflowing boxes. The eight-page
  output and the numerical model/settings table were visually inspected.
- Dataset selection tests retain every matching performance, including multiple
  performances from the same artist, preserve annotations, make dry runs without
  creating the destination, skip identical copies, and reject conflicting data.
  A real-source dry run found all 25 matching recordings and copied zero files.

## Deliberately not run

No Version 2 training loop, optimizer step, real-data feature preparation,
full-recording inference on the handpicked collection, GPU training, or accuracy
evaluation has been run. Tests constructing an optimizer only serialize an
untrained checkpoint to verify loading; they do not update model parameters.

Populate or expand `data/ver2/set` and the shared `data/test_ver1_handpicked` before running
the commands in the README. Existing Version 1 artifacts are preserved.

## Limits

The full decoded recording and resampled estimator input require RAM proportional
to duration. One recording-level estimate ensures consistent coordinates, but it
can still be wrong by an octave or another interval. A wrong estimate then affects
the entire recording. No improvement in recognition accuracy is claimed before
retraining and external evaluation. A recording with a changing tonic does not
fit the fixed-reference assumption.
