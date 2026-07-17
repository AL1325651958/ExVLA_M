# STVTA V17.3 Mask Regularization Design

## Goal

Create an independent V17.3 training version that reduces the Boom and Bucket
masks' tendency to occupy sky and distant-horizon shortcuts while preserving
the currently strong Swing branch and full V17.1 checkpoint compatibility.

## Root Cause

V17.1 flattens the eight masks (two modalities times four joints) and penalizes
every off-diagonal cosine similarity. This includes the RGB and elevation masks
for the same joint. Those two masks should be allowed to agree on a physically
meaningful region, but the current loss pushes one of them away. Together with
the coarse 14 x 14 grid and the mask-area lower bound, the cheapest alternative
can be a large, stable background structure such as the sky strip or mountain
horizon.

The defect is in the training regularizer, not in the V17.1 model tensor shapes.
The Swing decoder and its periodic/velocity supervision are already performing
well and must not be changed in V17.3.

## Version Boundary

V17.3 is a training-objective revision of V17.1, not a new network topology.
It uses the exact V17.1 model construction and state-dict layout. Therefore:

- V17.1 behavior remains the default when running `train_yolo_v17_1.py`.
- `train_yolo_v17_3.py` selects the V17.3 regularizer and metadata.
- V17.1 checkpoints can warm-start V17.3 with `--weights_only`.
- V17.3 checkpoints can be visualized by constructing the V17.1-compatible
  architecture while displaying the checkpoint version as V17.3.
- Checkpoint names and history names use a V17.3 prefix so V17.1 artifacts are
  never overwritten.

## V17.3 Diversity Loss

Let `M` have shape `[B, 2, 4, T, G, G]`, where dimension 1 is modality and
dimension 2 is joint. V17.3 normalizes every mask over its temporal-spatial
coordinates and computes a separate 4 x 4 joint-similarity matrix inside each
modality:

\[
s_{b,c,i,j} =
\frac{\langle M_{b,c,i}, M_{b,c,j}\rangle}
{\lVert M_{b,c,i}\rVert_2\lVert M_{b,c,j}\rVert_2 + \epsilon}.
\]

Only different-joint pairs in the same modality are regularized:

\[
\mathcal{L}_{\mathrm{div}}^{17.3} =
\frac{1}{2}\operatorname{mean}_{b,c,i\ne j}
\left[\max(0, s_{b,c,i,j}-0.5)^2\right].
\]

Consequently, same-joint RGB/elevation agreement is neither rewarded nor
penalized. The margin is relaxed from 0.3 to 0.5 so the loss discourages mask
collapse without forcing artificial spatial separation.

V17.1 retains its existing all-eight-mask loss and margin 0.3 for exact
reproducibility.

## Preserved Training Behavior

V17.3 does not change:

- the dual CSPDarknet/FPN-PAN towers or cross-modal attention;
- the independent RGB/elevation mask heads;
- the TemporalMaskMixer, Transformer encoder, or decoder layers;
- per-joint decoder coefficients, including Swing `lambda_m = 0` and
  `lambda_v = 0.5`;
- the Swing-weighted sin/cos loss, circular R-squared calculation, periodic pose
  auxiliary head, Swing velocity auxiliary head, EMA, or scheduler;
- the planar/Swing mask-area constraints and temporal smoothness term.

This one-variable change makes the V17.3 experiment directly comparable with
the V17.1 baseline.

## Mask Diagnostics

V17.3 reports validation-mask statistics without adding them to the loss:

- mean active area for every modality/joint mask;
- normalized center of mass `(x, y)` for every modality/joint mask, where
  `(0, 0)` is the upper-left grid corner;
- same-joint RGB/elevation cosine similarity.

The epoch log emphasizes Boom and Bucket, while the returned metric structure
contains all four joints. A persistently small `y` value exposes a sky shortcut;
high cross-modal similarity confirms that both modalities agree rather than one
being displaced only to satisfy the old diversity loss.

Diagnostics use detached tensors under validation and must not change gradients,
model outputs, checkpoint tensor shapes, or inference.

## Code Structure

- `vla_model/train_yolo_v17_1.py`
  - add pure, testable diversity and diagnostic helpers;
  - parameterize the training entry point with a V17.1/V17.3 variant;
  - keep V17.1 defaults unchanged.
- `vla_model/train_yolo_v17_3.py`
  - thin executable entry point selecting the V17.3 variant.
- `vla_model/visualize_yolo.py`
  - recognize V17.3 metadata and load it using the V17.1-compatible model
    architecture.
- `tests/test_v17_3_masks.py`
  - verify loss-pair selection, margin behavior, diagnostics, V17.1 defaults,
    V17.3 metadata, and visualization detection.

## Failure Handling

- Diversity and diagnostic helpers reject tensors not shaped
  `[B, 2, 4, T, G, G]` with a clear `ValueError`.
- An unknown training variant raises a `ValueError` before datasets or GPU
  objects are created.
- Existing `--weights_only` validation and compatible state loading remain the
  checkpoint migration path.

## Acceptance Criteria

1. Running `train_yolo_v17_1.py` preserves legacy diversity mode, margin,
   checkpoint metadata, and filenames.
2. Running `train_yolo_v17_3.py` uses within-modality different-joint diversity
   with margin 0.5 and saves V17.3 metadata/filenames.
3. Identical RGB/elevation masks for the same joint do not increase the V17.3
   diversity loss.
4. Identical masks for different joints in one modality do increase the V17.3
   diversity loss.
5. V17.3 validation logs expose Boom/Bucket area and center-of-mass statistics.
6. V17.1 weights load into V17.3 without model-shape migration.
7. The visualizer detects V17.3 checkpoints and loads all compatible tensors.
8. All existing V17.1 focused tests continue to pass.

