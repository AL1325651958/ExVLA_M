# All-Version Excavator Model Benchmark Design

## Goal

Create one server-side evaluation script that discovers the trained models
under `output/`, evaluates one representative checkpoint per run on the same
validation protocol, and incrementally exports comparable overall and
per-excavator MAE/R-squared metrics to CSV.

## Evaluation Protocol

Every checkpoint is evaluated with the same data policy:

- dataset root: `data/excavator-motion`;
- split: validation, using the dataset's deterministic seed-42 stratified split;
- `train_split=0.857`;
- `sample_ratio=0.2`;
- exclude excavator ID 1 (306 nighttime data);
- include excavator ID 0 (75) and ID 2 (490);
- pure-visual inference: no joint-state tensor is supplied to the model;
- models are evaluated sequentially and GPU objects are released between runs.

The evaluator may reconstruct separate validation datasets when checkpoints
require different `seq_len` or `img_size` values. Results record these values so
comparisons with different input windows remain explicit.

## Checkpoint Discovery and Selection

The script recursively discovers `.pt` files below `--output_root` and groups
them by their direct parent directory. Each directory contributes at most one
checkpoint.

Selection priority is:

1. an overall-best checkpoint whose filename contains `best` but not
   `best_swing`;
2. otherwise, the checkpoint with the largest parsed `epoch_<N>` number;
3. otherwise, the directory is recorded as skipped with a selection error.

Backbone-only and export artifacts are never benchmarked. Filenames containing
`backbone_pretrained`, `pretrain`, or `optimizer` are excluded. If a directory
contains multiple overall-best candidates, the most recently modified candidate
is selected and the decision is printed.

This policy deliberately excludes `best_swing`: the benchmark selects the best
overall action model, not a checkpoint optimized for only one joint.

## Model Adapter Registry

The evaluator uses an ordered adapter registry. Each adapter has three clear
responsibilities: determine whether it recognizes a checkpoint, construct the
matching model from checkpoint metadata/state shapes, and convert the model's
raw output to four joint angles in radians.

### YOLO/STVTA grid family

Supports checkpoints handled by `vla_model.model_yolo`, including legacy
YOLO-STVTA signatures and V9 through V17.3. It reuses version detection,
Transformer-dimension inference, legacy key remapping, V17.1 mask migration,
and compatible state loading. V17.3 metadata maps to the V17.1-compatible model
topology.

### Dual-tower STVTA family

Supports `vla_model.model_stvta` checkpoints from V12 through V15. Model width,
encoder depth, feed-forward width, number of excavator heads, sequence length,
and image size are recovered from checkpoint config and state shapes. Raw
sin/cos action pairs are decoded with the model's `decode_action` method.

### Legacy VLA family

Supports checkpoints built with `vla_model.model`, including Transformer and
Mamba encoders and linear or sin/cos output variants when their state signature
and stored config are sufficient to reconstruct the architecture. Evaluation
always calls the model in `eval()` mode with `qpos=None`.

If no adapter recognizes a checkpoint, or compatible loading would omit a
required tensor, the evaluator writes a `status=error` row and continues. It
does not silently report metrics from a partially initialized model.

## Prediction and Angular Metrics

All predictions and targets have shape `[N, 4]` in radians and joint order:

1. Boom (large arm)
2. Arm (stick)
3. Bucket
4. Swing

Boom, Arm, and Bucket use linear angular residuals. Swing uses a wrapped
residual:

\[
e_i = \operatorname{atan2}
\left(\sin(\hat\theta_i-\theta_i),
      \cos(\hat\theta_i-\theta_i)\right).
\]

Per-joint MAE is `mean(abs(e))`. For the three planar joints, R-squared uses the
ordinary target mean. Swing R-squared uses the circular target mean and wrapped
target deviations:

\[
R^2_{\mathrm{swing}} = 1 -
\frac{\sum_i e_i^2}
     {\sum_i \operatorname{wrap}(\theta_i-\bar\theta_{\mathrm{circ}})^2}.
\]

`mae_mean` and `r2_mean` are arithmetic means of the four per-joint values.
Metrics are computed for three scopes: `overall`, excavator `75`, and excavator
`490`.

## CSV Schema

Each successfully evaluated model produces three rows. A failed or skipped
model produces one row with identifying fields, `status`, and `error` populated.

Columns are:

```text
model_dir
checkpoint
detected_family
detected_version
checkpoint_epoch
status
scope
excavator_id
excavator_name
n_samples
seq_len
img_size
loaded_tensors
skipped_tensors
mae_mean
r2_mean
boom_mae
boom_r2
arm_mae
arm_r2
bucket_mae
bucket_r2
swing_mae
swing_r2
elapsed_seconds
error
```

The CSV is opened once, receives its header immediately, and is flushed after
every model. A crash or manual interruption therefore preserves completed
results.

## Command-Line Interface

The primary invocation is:

```bash
python scripts/evaluate_all_models.py \
  --output_root output \
  --data_dir data/excavator-motion \
  --sample_ratio 0.2 \
  --exclude_excavators 1 \
  --batch_size 8 \
  --csv output/all_model_metrics.csv
```

Additional options provide `--device`, `--num_workers`, `--train_split`, and
fallback `--seq_len`/`--img_size` values for old checkpoints without config.
Defaults remain conservative for server GPU memory.

## Progress and Resource Management

The outer progress bar reports checkpoint count and selected model directory.
The inner progress bar reports validation batches. After each checkpoint the
script deletes model/prediction tensors, runs Python garbage collection, and
empties the CUDA cache. Only one model is resident on the GPU at a time.

Validation datasets and data loaders are cached by
`(seq_len, img_size, sample_ratio, excluded_ids)` so versions with matching
inputs do not reload HDF5 metadata repeatedly.

## Failure Handling

- Bad/corrupt checkpoints produce an error row and console traceback summary.
- Backbone-only checkpoints are filtered before loading.
- Unknown architecture signatures produce an error row.
- A checkpoint with missing required model tensors is rejected rather than
  evaluated with random weights.
- An empty `overall`, `75`, or `490` scope is omitted; the console reports it.
- `KeyboardInterrupt` flushes the CSV before propagating, allowing a clean
  restart.

## Testing

Unit tests cover:

- best-overall selection and `best_swing` exclusion;
- latest-epoch fallback;
- backbone-only exclusion;
- wrapped Swing MAE/R-squared across the `-pi/+pi` boundary;
- ordinary planar joint metrics;
- overall/75/490 row generation;
- CSV error rows;
- adapter selection using synthetic state dictionaries;
- pure-visual forwarding (`qpos=None`);
- rejection of incomplete compatible loads.

A lightweight integration test evaluates a tiny synthetic checkpoint and
dataset without requiring the real server data.

## Acceptance Criteria

1. One command scans `output/` and selects at most one overall-best model per
   checkpoint directory.
2. 306 data is excluded and validation sampling is fixed at 0.2.
3. Every successful model produces comparable overall, 75, and 490 metrics.
4. CSV contains mean and per-joint MAE/R-squared values.
5. Swing metrics are circular and correct across angle wrap boundaries.
6. Evaluation remains pure visual for every adapter.
7. Models run sequentially without accumulating GPU memory.
8. Unsupported checkpoints do not terminate the batch and are visible as CSV
   error rows.
9. Completed CSV rows survive interruption.

