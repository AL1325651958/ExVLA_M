# V10 Pure-Visual Temporal Masks

## Goal

Make each joint mask respond to visual temporal evidence while keeping inference
strictly vision-only. Deployment inputs remain RGB frames, elevation frames, and
the excavator type; qpos must not affect the V10 forward pass.

## Architecture

1. Build spatial visual tokens from the existing RGB and elevation towers.
2. Apply a lightweight temporal mixer independently at each grid location across
   the input sequence. The mixed tokens retain one representation per frame and
   spatial cell.
3. Generate a separate mask for each joint from these temporally mixed visual
   tokens and the joint embedding. No qpos-derived tensor is provided to this
   branch.
4. Continue using the masks to gate the encoder and bias the per-joint decoder
   cross-attention.

## Training-only qpos supervision

qpos remains available in batches but is used only to compute an auxiliary loss.
A small training-only pose head reads the temporally mixed visual representation
and predicts the final observed qpos in the input window. Its loss encourages the
visual representation used by masks to preserve pose information. The pose head
and its loss are not used during validation or inference.

## Visualization

The visualizer will render the last frame's raw `masks_spatial` by default.
An explicit `--mask_view avg` option preserves the previous eight-frame average
view for comparison.

## Compatibility and validation

The V10 training entry point and output directory are separate from V9. V10
checkpoints include the new temporal mixer and training-only head. Tests verify
that the model output is invariant to qpos, that the raw last-frame mask view is
selected correctly, and that the average view remains available.
