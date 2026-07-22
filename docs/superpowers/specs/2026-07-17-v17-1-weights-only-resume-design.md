# V17.1 Weights-Only Resume Design

## Goal

Allow a V17.1 best checkpoint to initialize a new low-learning-rate training run without restoring its epoch, optimizer, scaler, scheduler, or historical best metrics.

## Interface

`train_yolo_v17_1.py` gains `--weights_only`. It is valid only together with `--resume CHECKPOINT`.

- Normal `--resume` remains unchanged and performs a full training-state continuation.
- `--resume ... --weights_only` loads `model_state_dict`, which contains the checkpoint's evaluated EMA weights.
- Weights-only mode starts at epoch 0, leaves the newly constructed optimizer/scaler/scheduler untouched, and resets best loss and best Swing R2.
- Existing compatible-state loading and legacy V17.1 mask migration remain active.

## Failure Handling

Using `--weights_only` without `--resume` is rejected by argument parsing. Shape-incompatible tensors continue to be skipped and reported by the existing compatible checkpoint loader.

## Testing

Regression tests verify that weights-only mode selects EMA rather than raw weights, resets training metadata, and does not request training-state restoration. A second test verifies that ordinary resume keeps its existing raw-weight and metadata behavior.
