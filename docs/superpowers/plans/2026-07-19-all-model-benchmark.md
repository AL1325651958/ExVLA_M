# All-Version Excavator Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fault-tolerant server evaluator that selects one overall-best checkpoint per output directory, evaluates compatible model families on the same 75/490 validation subset, and incrementally exports overall and per-joint angular metrics to CSV.

**Architecture:** Put reusable discovery, metric, adapter, and evaluation logic in `vla_model/benchmark.py`; keep `scripts/evaluate_all_models.py` as a thin CLI. Adapters reconstruct a model only when all tensors required by its current forward path are present, preventing invalid partial-random evaluations.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `ExcavatorDataset`, `tqdm`, standard-library `csv`, unittest/pytest.

---

## File Map

- Create `vla_model/benchmark.py`: checkpoint selection, angular metrics, adapters, evaluation, and CSV rows.
- Create `scripts/evaluate_all_models.py`: CLI and sequential orchestration.
- Modify `vla_model/visualize_yolo.py`: expose existing legacy state normalization for reuse.
- Create `tests/test_model_benchmark.py`: unit and lightweight integration tests.

### Task 1: Checkpoint selection and angular metrics

**Files:**
- Create: `vla_model/benchmark.py`
- Create: `tests/test_model_benchmark.py`

- [ ] **Step 1: Write failing selection tests**

```python
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vla_model.benchmark import discover_checkpoints


class CheckpointSelectionTests(unittest.TestCase):
    def test_prefers_overall_best_and_excludes_best_swing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "V17_3"
            run.mkdir()
            best = run / "yolo_v17_3_checkpoint_best.pt"
            best.write_bytes(b"best")
            (run / "yolo_v17_3_checkpoint_best_swing.pt").write_bytes(b"swing")
            (run / "yolo_v17_3_checkpoint_epoch_40.pt").write_bytes(b"epoch")
            selected, skipped = discover_checkpoints(Path(tmp))
        self.assertEqual([item.checkpoint.name for item in selected], [best.name])
        self.assertEqual(skipped, [])

    def test_uses_largest_epoch_when_best_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "V16"
            run.mkdir()
            (run / "yolo_v16_checkpoint_epoch_10.pt").write_bytes(b"10")
            latest = run / "yolo_v16_checkpoint_epoch_80.pt"
            latest.write_bytes(b"80")
            selected, _ = discover_checkpoints(Path(tmp))
        self.assertEqual(selected[0].checkpoint.name, latest.name)

    def test_ignores_backbone_pretraining_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "checkpoints"
            run.mkdir()
            (run / "yolo_backbone_pretrained.pt").write_bytes(b"stub")
            selected, skipped = discover_checkpoints(Path(tmp))
        self.assertEqual(selected, [])
        self.assertEqual(len(skipped), 1)
```

- [ ] **Step 2: Run selection tests and verify RED**

Run: `python -m pytest tests/test_model_benchmark.py::CheckpointSelectionTests -q`

Expected: collection fails because `vla_model.benchmark` does not exist.

- [ ] **Step 3: Implement selection**

```python
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SelectedCheckpoint:
    model_dir: Path
    checkpoint: Path
    reason: str


_EXCLUDED_NAMES = ("backbone_pretrained", "pretrain", "optimizer")


def _epoch_number(path):
    match = re.search(r"epoch_(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def discover_checkpoints(output_root):
    output_root = Path(output_root)
    every_file = list(output_root.rglob("*.pt"))
    groups = {}
    for path in every_file:
        if any(token in path.name.lower() for token in _EXCLUDED_NAMES):
            continue
        groups.setdefault(path.parent, []).append(path)
    selected, skipped = [], []
    for directory in sorted({path.parent for path in every_file}):
        candidates = groups.get(directory, [])
        best = [path for path in candidates
                if "best" in path.stem.lower()
                and "best_swing" not in path.stem.lower()]
        if best:
            winner = max(best, key=lambda path: (path.stat().st_mtime, path.name))
            selected.append(SelectedCheckpoint(directory, winner, "overall_best"))
            continue
        epochs = [path for path in candidates if _epoch_number(path) >= 0]
        if epochs:
            winner = max(epochs, key=lambda path: (_epoch_number(path), path.name))
            selected.append(SelectedCheckpoint(directory, winner, "latest_epoch"))
        else:
            skipped.append((directory, "no overall-best or epoch checkpoint"))
    return selected, skipped
```

- [ ] **Step 4: Write failing angular-metric tests**

```python
from vla_model.benchmark import compute_angular_metrics


class AngularMetricTests(unittest.TestCase):
    def test_swing_wraps_across_pi_boundary(self):
        target = np.array([[0., 0., 0., math.pi - .05],
                           [1., 1., 1., -math.pi + .05]])
        prediction = target.copy()
        prediction[:, 3] *= -1
        metrics = compute_angular_metrics(prediction, target)
        self.assertLess(metrics["mae"][3], .11)

    def test_planar_metrics_stay_linear(self):
        target = np.array([[0., 0., 0., 0.], [2., 2., 2., 1.]])
        prediction = target.copy()
        prediction[:, 0] += 1
        metrics = compute_angular_metrics(prediction, target)
        self.assertAlmostEqual(metrics["mae"][0], 1.)
        self.assertAlmostEqual(metrics["r2"][0], 0.)
```

- [ ] **Step 5: Verify RED, implement metrics, and verify GREEN**

Run: `python -m pytest tests/test_model_benchmark.py::AngularMetricTests -q`

Implement:

```python
def circular_delta(prediction, target):
    delta = prediction - target
    return np.arctan2(np.sin(delta), np.cos(delta))


def _safe_r2(residual, centered_target):
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(centered_target).sum())
    if ss_tot <= 1e-10:
        return 1.0 if ss_res <= 1e-10 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_angular_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("prediction and target must both have shape [N, 4]")
    if len(target) == 0:
        raise ValueError("at least one sample is required")
    residual = prediction - target
    residual[:, 3] = circular_delta(prediction[:, 3], target[:, 3])
    mae = np.abs(residual).mean(axis=0)
    r2 = np.zeros(4)
    for joint in range(3):
        r2[joint] = _safe_r2(residual[:, joint], target[:, joint] - target[:, joint].mean())
    swing_mean = np.arctan2(np.sin(target[:, 3]).mean(), np.cos(target[:, 3]).mean())
    r2[3] = _safe_r2(residual[:, 3], circular_delta(target[:, 3], swing_mean))
    return {"n_samples": len(target), "mae": mae.tolist(), "mae_mean": float(mae.mean()),
            "r2": r2.tolist(), "r2_mean": float(r2.mean())}
```

Run: `python -m pytest tests/test_model_benchmark.py -q`

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- vla_model/benchmark.py tests/test_model_benchmark.py
git commit -m "feat: add benchmark selection and metrics"
```

### Task 2: Shared YOLO normalization and strict model loading

**Files:**
- Modify: `vla_model/visualize_yolo.py`
- Modify: `vla_model/benchmark.py`
- Modify: `tests/test_model_benchmark.py`

- [ ] **Step 1: Write failing helper tests**

```python
import torch
from torch import nn
from vla_model.benchmark import load_complete_state_dict
from vla_model.visualize_yolo import normalize_yolo_state_dict


class AdapterFoundationTests(unittest.TestCase):
    def test_normalizes_legacy_shared_masks(self):
        state = {"mask_linear1.weight": torch.ones(2, 3),
                 "mask_linear1.bias": torch.ones(2),
                 "mask_linear2.weight": torch.ones(1, 2),
                 "mask_linear2.bias": torch.ones(1)}
        self.assertIn("mask_linear1.0.weight", normalize_yolo_state_dict(state))

    def test_rejects_incomplete_target_state(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        incomplete = {"0.weight": model.state_dict()["0.weight"]}
        with self.assertRaisesRegex(ValueError, "missing required tensors"):
            load_complete_state_dict(model, incomplete)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_model_benchmark.py::AdapterFoundationTests -q`

- [ ] **Step 3: Extract `normalize_yolo_state_dict`**

Move the existing V17.1 migration, V5 action-head remap, legacy mask-head remap,
and `query_tokens` rename from `visualize_yolo.main()` into a public pure
function. Make the visualizer call that function so behavior remains unchanged.

```python
def normalize_yolo_state_dict(state_dict):
    """Return a normalized copy without mutating the checkpoint payload."""
    normalized = upgrade_legacy_v17_1_state_dict(dict(state_dict))
    keys = set(normalized)
    is_v5 = (any("action_heads" in key for key in keys)
             and not any("joint_embed" in key for key in keys))
    remapped = {}
    for key, value in list(normalized.items()):
        if is_v5 and "action_heads." in key:
            parts = key.split(".")
            excavator_id = int(parts[1])
            rest = ".".join(parts[2:])
            if not (rest.startswith("6.") or rest.startswith("3.")):
                for joint in range(4):
                    remapped[f"action_heads.{excavator_id}.{joint}.{rest}"] = value.clone()
        if "mask_head" in key and "mask_heads" not in key:
            for joint in range(4):
                remapped[key.replace("mask_head", f"mask_heads.{joint}")] = value.clone()
    normalized.update(remapped)
    if "query_tokens" in normalized and "joint_queries" not in normalized:
        normalized["joint_queries"] = normalized.pop("query_tokens")
    return normalized
```

In `main()`, replace the inline remap block with
`state_dict = normalize_yolo_state_dict(ckpt["model_state_dict"])`, then derive
`sd_keys` and call `detect_version`. Calling the V17.1 upgrader unconditionally
is safe because it is a no-op when legacy shared mask tensors are absent.

- [ ] **Step 4: Implement strict compatible loading**

```python
def load_complete_state_dict(model, state_dict):
    target = model.state_dict()
    compatible = {key: value for key, value in state_dict.items()
                  if key in target and target[key].shape == value.shape}
    missing = sorted(set(target) - set(compatible))
    if missing:
        raise ValueError(f"missing required tensors ({len(missing)}): {', '.join(missing[:5])}")
    model.load_state_dict(compatible, strict=True)
    return len(compatible), len(state_dict) - len(compatible)
```

- [ ] **Step 5: Run regressions and commit**

```powershell
python -m pytest tests/test_model_benchmark.py tests/test_v17_1_swing.py tests/test_v17_3_masks.py -q
git add -- vla_model/visualize_yolo.py vla_model/benchmark.py tests/test_model_benchmark.py
git commit -m "refactor: share YOLO checkpoint normalization"
```

### Task 3: Model adapters and pure-visual prediction

**Files:**
- Modify: `vla_model/benchmark.py`
- Modify: `tests/test_model_benchmark.py`

- [ ] **Step 1: Write failing routing and qpos tests**

```python
from unittest import mock
from vla_model.benchmark import LoadedBenchmarkModel, choose_adapter_family


class AdapterRoutingTests(unittest.TestCase):
    def test_routes_three_families(self):
        self.assertEqual(choose_adapter_family({"rgb_backbone.stem.conv.weight"}), "yolo")
        self.assertEqual(choose_adapter_family({"rgb_branch.backbone.stem.conv.weight"}), "stvta")
        self.assertEqual(choose_adapter_family({"vision_encoder.rgb_backbone.conv1.weight"}), "legacy")

    def test_prediction_passes_none_for_qpos(self):
        model = mock.Mock()
        model.return_value = (torch.tensor([[0., 1.] * 4]), None, None)
        model.decode_action.return_value = torch.zeros(1, 4)
        loaded = LoadedBenchmarkModel(model, "yolo", "V17.3", 8, 224, 1, 0,
                                      "interleaved_sincos", 1)
        loaded.predict(torch.zeros(1, 8, 3, 4, 4),
                       torch.zeros(1, 8, 3, 4, 4), torch.zeros(1, dtype=torch.long))
        self.assertIsNone(model.call_args.args[2])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_model_benchmark.py::AdapterRoutingTests -q`

- [ ] **Step 3: Implement adapter contract and routing**

```python
@dataclass
class LoadedBenchmarkModel:
    model: object
    family: str
    version: str
    seq_len: int
    img_size: int
    loaded_tensors: int
    skipped_tensors: int
    output_encoding: str
    action_chunk: int = 1

    @torch.no_grad()
    def predict(self, rgb, elevation, excavator_id):
        output = self.model(rgb, elevation, None, excavator_id)
        raw = output[0] if isinstance(output, tuple) else output
        if raw.ndim == 3:
            raw = raw[:, 0]
        if self.output_encoding == "interleaved_sincos":
            return self.model.decode_action(raw)
        if self.output_encoding == "split_sincos":
            sin, cos = raw.chunk(2, dim=-1)
            return torch.atan2(sin, cos)
        if self.output_encoding == "radians":
            return raw
        raise ValueError(f"unknown output encoding: {self.output_encoding}")


def choose_adapter_family(state_keys):
    if any(key.startswith("rgb_branch.") for key in state_keys):
        return "stvta"
    if any(key.startswith("rgb_backbone.") for key in state_keys):
        return "yolo"
    if any(key.startswith("vision_encoder.") for key in state_keys):
        return "legacy"
    raise ValueError("unsupported checkpoint architecture")
```

- [ ] **Step 4: Implement `load_benchmark_model` and three builders**

Add configuration and state-shape helpers:

```python
def _config_value(checkpoint, name, default=None, required=False):
    config = checkpoint.get("config")
    if isinstance(config, dict):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default) if config is not None else default
    if required and value is None:
        raise ValueError(f"checkpoint config is missing {name}")
    return value


def _layer_count(state, prefix):
    indices = {
        int(key[len(prefix):].split(".", 1)[0])
        for key in state if key.startswith(prefix)
    }
    if not indices:
        raise ValueError(f"cannot infer layers from {prefix}")
    return max(indices) + 1


def _num_excavators(state):
    weight = state.get("excv_embed.weight")
    if weight is None:
        raise ValueError("checkpoint has no excavator embedding")
    return int(weight.shape[0])
```

Implement the three builders. Head counts are required from stored config
because tensor shapes do not encode the number of attention heads; silently
guessing them would change inference semantics.

```python
def _build_yolo(checkpoint, state, fallback_seq_len, fallback_img_size):
    from vla_model.model_yolo import ExcavatorVLAYolo
    from vla_model.visualize_yolo import (
        detect_version, infer_transformer_config, normalize_yolo_state_dict,
    )

    state = normalize_yolo_state_dict(state)
    version_tag, _, model_version = detect_version(
        set(state), checkpoint.get("model_version"))
    hidden_dim, n_layers, ff_dim = infer_transformer_config(state)
    seq_len = int(_config_value(checkpoint, "seq_len", fallback_seq_len))
    img_size = int(_config_value(checkpoint, "img_size", fallback_img_size))
    n_heads = int(_config_value(checkpoint, "n_heads", required=True))
    model = ExcavatorVLAYolo(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
        dropout=float(_config_value(checkpoint, "dropout", 0.0)),
        pretrained=False, num_excavators=_num_excavators(state),
        version=model_version,
    )
    loaded, skipped = load_complete_state_dict(model, state)
    return LoadedBenchmarkModel(
        model, "yolo", version_tag, seq_len, img_size, loaded, skipped,
        "interleaved_sincos", 1)


def _build_stvta(checkpoint, state, fallback_seq_len, fallback_img_size):
    from vla_model.model_stvta import ExcavatorSTVTA

    hidden_dim = int(state["excv_embed.weight"].shape[1])
    n_layers = _layer_count(state, "rgb_branch.encoder.layers.")
    ff_dim = int(state["rgb_branch.encoder.layers.0.linear1.weight"].shape[0])
    seq_len = int(_config_value(checkpoint, "seq_len", fallback_seq_len))
    img_size = int(_config_value(checkpoint, "img_size", fallback_img_size))
    n_heads = int(_config_value(checkpoint, "n_heads", required=True))
    version = str(checkpoint.get("model_version") or "stvta")
    model = ExcavatorSTVTA(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
        dropout=float(_config_value(checkpoint, "dropout", 0.0)),
        pretrained=False, num_excavators=_num_excavators(state), version=version,
    )
    loaded, skipped = load_complete_state_dict(model, state)
    return LoadedBenchmarkModel(
        model, "stvta", version.upper(), seq_len, img_size, loaded, skipped,
        "interleaved_sincos", 1)


def _build_legacy(checkpoint, state, fallback_seq_len, fallback_img_size):
    from vla_model.model import ExcavatorVLA

    required = lambda name: _config_value(checkpoint, name, required=True)
    use_sincos_output = bool(required("use_sincos_output"))
    action_chunk = int(_config_value(checkpoint, "action_chunk", 1))
    seq_len = int(_config_value(checkpoint, "seq_len", fallback_seq_len))
    img_size = int(_config_value(checkpoint, "img_size", fallback_img_size))
    model = ExcavatorVLA(
        seq_len=seq_len,
        hidden_dim=int(required("hidden_dim")),
        n_heads=int(required("n_heads")),
        n_layers=int(required("n_layers")),
        ff_dim=int(required("ff_dim")),
        dropout=float(_config_value(checkpoint, "dropout", 0.0)),
        drop_path_rate=float(_config_value(checkpoint, "drop_path_rate", 0.0)),
        pretrained=False,
        num_excavators=_num_excavators(state),
        qpos_mode=str(_config_value(checkpoint, "qpos_mode", "none")),
        qpos_drop_prob=0.0,
        encoder_type=str(_config_value(checkpoint, "encoder_type", "transformer")),
        mamba_d_state=int(_config_value(checkpoint, "mamba_d_state", 0)),
        mamba_d_conv=int(_config_value(checkpoint, "mamba_d_conv", 4)),
        mamba_expand=int(_config_value(checkpoint, "mamba_expand", 2)),
        use_sincos=bool(_config_value(checkpoint, "use_sincos", False)),
        use_sincos_output=use_sincos_output,
        action_chunk=action_chunk,
    )
    loaded, skipped = load_complete_state_dict(model, state)
    encoding = "split_sincos" if use_sincos_output else "radians"
    return LoadedBenchmarkModel(
        model, "legacy", "legacy", seq_len, img_size,
        loaded, skipped, encoding, action_chunk)


def load_benchmark_model(path, device, fallback_seq_len=8, fallback_img_size=224):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    family = choose_adapter_family(set(state))
    builders = {"yolo": _build_yolo, "stvta": _build_stvta,
                "legacy": _build_legacy}
    loaded = builders[family](checkpoint, state, fallback_seq_len,
                              fallback_img_size)
    loaded.model.to(device).eval()
    return loaded, int(checkpoint.get("epoch", -1))
```

- [ ] **Step 5: Add exact-load synthetic adapter tests**

Keep these tests lightweight by mocking the family builders rather than
allocating three real CNN towers. Verify the public loader dispatches, moves to
the requested device, calls `eval()`, and returns the stored epoch:

```python
    @mock.patch("vla_model.benchmark._build_yolo")
    @mock.patch("vla_model.benchmark.torch.load")
    def test_public_loader_dispatches_yolo(self, torch_load, build_yolo):
        torch_load.return_value = {
            "epoch": 17,
            "model_state_dict": {"rgb_backbone.stem.conv.weight": torch.ones(1)},
        }
        fake_model = mock.Mock()
        expected = LoadedBenchmarkModel(
            fake_model, "yolo", "V17.3", 8, 224, 1, 0,
            "interleaved_sincos", 1)
        build_yolo.return_value = expected
        loaded, epoch = load_benchmark_model("model.pt", "cpu")
        self.assertIs(loaded, expected)
        self.assertEqual(epoch, 17)
        fake_model.to.assert_called_once_with("cpu")
        fake_model.eval.assert_called_once_with()
```

The strict completeness behavior itself is already covered by
`AdapterFoundationTests`; a server smoke run in Task 5 exercises real
checkpoints without duplicating multi-gigabyte model allocation in unit tests.

- [ ] **Step 6: Run and commit**

```powershell
python -m pytest tests/test_model_benchmark.py -q
git add -- vla_model/benchmark.py tests/test_model_benchmark.py
git commit -m "feat: add benchmark model adapters"
```

### Task 4: Scoped rows, evaluation runner, and CSV

**Files:**
- Modify: `vla_model/benchmark.py`
- Modify: `tests/test_model_benchmark.py`

- [ ] **Step 1: Write failing row test**

```python
from vla_model.benchmark import build_metric_rows


class MetricRowTests(unittest.TestCase):
    def test_builds_overall_75_and_490_rows(self):
        prediction = np.zeros((4, 4))
        target = np.zeros((4, 4))
        ids = np.array([0, 0, 2, 2])
        rows = build_metric_rows({"model_dir": "V17_3", "checkpoint": "best.pt"},
                                 prediction, target, ids)
        self.assertEqual([row["scope"] for row in rows], ["overall", "75", "490"])
        self.assertEqual(rows[1]["n_samples"], 2)
        self.assertIn("swing_r2", rows[0])
```

- [ ] **Step 2: Verify RED and implement rows**

Run: `python -m pytest tests/test_model_benchmark.py::MetricRowTests -q`

Use this exact implementation:

```python
JOINT_NAMES = ("boom", "arm", "bucket", "swing")


def build_metric_rows(identity, prediction, target, excavator_ids):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    excavator_ids = np.asarray(excavator_ids)
    rows = []
    for scope, excavator_id, excavator_name in (
        ("overall", None, "75+490"), ("75", 0, "75"), ("490", 2, "490"),
    ):
        keep = np.ones(len(target), dtype=bool)
        if excavator_id is not None:
            keep = excavator_ids == excavator_id
        if not keep.any():
            continue
        metrics = compute_angular_metrics(prediction[keep], target[keep])
        row = dict(identity)
        row.update({
            "status": "ok", "scope": scope,
            "excavator_id": "" if excavator_id is None else excavator_id,
            "excavator_name": excavator_name,
            "n_samples": metrics["n_samples"],
            "mae_mean": metrics["mae_mean"],
            "r2_mean": metrics["r2_mean"], "error": "",
        })
        for joint, name in enumerate(JOINT_NAMES):
            row[f"{name}_mae"] = metrics["mae"][joint]
            row[f"{name}_r2"] = metrics["r2"][joint]
        rows.append(row)
    return rows
```

Implement `build_metric_rows` with scopes
`("overall", None)`, `("75", 0)`, and `("490", 2)`. Flatten the four-element
MAE/R² arrays into `boom_*`, `arm_*`, `bucket_*`, and `swing_*` columns.

- [ ] **Step 3: Implement `evaluate_loaded_model`**

Under `torch.inference_mode()`, move only RGB, elevation, and excavator IDs to
the selected device, call `loaded.predict`, select target
`batch["action"][:, 0]`, and concatenate CPU NumPy prediction/target/ID arrays.
Use an inner `tqdm` progress bar.

```python
def evaluate_loaded_model(loaded, data_loader, device, description="evaluate"):
    predictions, targets, excavator_ids = [], [], []
    loaded.model.eval()
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=description, leave=False):
            rgb = batch["rgb"].to(device, non_blocking=True)
            elevation = batch["elevation"].to(device, non_blocking=True)
            ids = batch["excavator_id"].to(device, non_blocking=True).long()
            prediction = loaded.predict(rgb, elevation, ids)
            target = batch["action"][:, 0]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
            if not torch.isfinite(prediction).all():
                raise ValueError("prediction contains NaN or Inf")
            predictions.append(prediction.float().cpu().numpy())
            targets.append(target.float().cpu().numpy())
            excavator_ids.append(ids.cpu().numpy())
    if not targets:
        raise ValueError("validation loader produced no samples")
    return (np.concatenate(predictions), np.concatenate(targets),
            np.concatenate(excavator_ids))
```

- [ ] **Step 4: Implement CSV schema and failure rows**

Define the exact columns from the design spec. `make_error_row` fills identity,
status, elapsed time, and a one-line error. `write_rows` calls `writer.writerow`
and `file.flush()` after every row.

```python
CSV_COLUMNS = [
    "model_dir", "checkpoint", "selection_reason", "detected_family",
    "detected_version", "checkpoint_epoch", "status", "scope",
    "excavator_id", "excavator_name", "n_samples", "seq_len", "img_size",
    "loaded_tensors", "skipped_tensors", "mae_mean", "r2_mean",
    "boom_mae", "boom_r2", "arm_mae", "arm_r2", "bucket_mae",
    "bucket_r2", "swing_mae", "swing_r2", "elapsed_seconds", "error",
]


def make_error_row(identity, error, elapsed_seconds):
    row = {column: "" for column in CSV_COLUMNS}
    row.update(identity)
    row.update({
        "status": "error", "scope": "error",
        "elapsed_seconds": elapsed_seconds,
        "error": " ".join(str(error).splitlines()),
    })
    return row


def write_rows(writer, file_handle, rows):
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
        file_handle.flush()
```

Add a test using `io.StringIO` that calls `write_rows` with one success and one
error row, re-reads them with `csv.DictReader`, and asserts both statuses and
the `swing_mae` column survive serialization.

- [ ] **Step 5: Run and commit**

```powershell
python -m pytest tests/test_model_benchmark.py -q
git add -- vla_model/benchmark.py tests/test_model_benchmark.py
git commit -m "feat: add benchmark evaluation and CSV rows"
```

### Task 5: CLI, sequential cleanup, and final verification

**Files:**
- Create: `scripts/evaluate_all_models.py`
- Modify: `tests/test_model_benchmark.py`

- [ ] **Step 1: Write failing CLI-default test**

```python
from scripts.evaluate_all_models import build_parser


class BenchmarkCliTests(unittest.TestCase):
    def test_defaults_match_approved_protocol(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.sample_ratio, .2)
        self.assertEqual(args.exclude_excavators, [1])
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.csv, "output/all_model_metrics.csv")
```

- [ ] **Step 2: Verify RED and implement parser**

Run: `python -m pytest tests/test_model_benchmark.py::BenchmarkCliTests -q`

Parser defaults: output root `output`, data root `data/excavator-motion`, ratio
`.2`, exclusion `[1]`, batch size `8`, CSV `output/all_model_metrics.csv`, train
split `.857`, workers `0`, fallback seq length `8`, fallback image size `224`,
and CUDA when available.

```python
def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one representative checkpoint per model directory")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--data_dir", default="data/excavator-motion")
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--exclude_excavators", type=int, nargs="*", default=[1])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--csv", default="output/all_model_metrics.csv")
    parser.add_argument("--train_split", type=float, default=0.857)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fallback_seq_len", type=int, default=8)
    parser.add_argument("--fallback_img_size", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser
```

- [ ] **Step 3: Implement cached loader factory and orchestration**

Cache `ExcavatorDataset`/`DataLoader` by `(seq_len, img_size)`. Construct the
validation dataset with `action_chunk=1`, `split="val"`, configured train split,
ratio, and excluded IDs. Open the CSV once, write/flush its header, and loop over
selected checkpoints with an outer progress bar.

For each checkpoint: load adapter, get loader, evaluate, write three success
rows; on ordinary exceptions write one error row and continue; in `finally`
delete model references, call `gc.collect()`, and empty the CUDA cache. Allow
`KeyboardInterrupt` to propagate after the file context flushes.

Implement `scripts/evaluate_all_models.py` with this orchestration. Keep
`run_benchmark` separate so the integration test can replace heavyweight
dependencies without launching real inference:

```python
import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vla_model.benchmark import (
    CSV_COLUMNS, build_metric_rows, discover_checkpoints,
    evaluate_loaded_model, load_benchmark_model, make_error_row, write_rows,
)
from vla_model.dataset import ExcavatorDataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one representative checkpoint per model directory")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--data_dir", default="data/excavator-motion")
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--exclude_excavators", type=int, nargs="*", default=[1])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--csv", default="output/all_model_metrics.csv")
    parser.add_argument("--train_split", type=float, default=0.857)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fallback_seq_len", type=int, default=8)
    parser.add_argument("--fallback_img_size", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def make_loader_factory(args):
    cache = {}

    def get_loader(seq_len, img_size):
        key = (int(seq_len), int(img_size))
        if key not in cache:
            dataset = ExcavatorDataset(
                data_dir=args.data_dir, seq_len=key[0], action_chunk=1,
                img_size=key[1], split="val", train_split=args.train_split,
                sample_ratio=args.sample_ratio,
                exclude_excv=set(args.exclude_excavators),
            )
            cache[key] = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
            )
        return cache[key]

    return get_loader


def _relative(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def run_benchmark(args):
    output_root = Path(args.output_root).resolve()
    selected, skipped = discover_checkpoints(output_root)
    for directory, reason in skipped:
        print(f"Skip {directory}: {reason}")
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    get_loader = make_loader_factory(args)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        file_handle.flush()
        for item in tqdm(selected, desc="models"):
            started = time.perf_counter()
            loaded = None
            identity = {
                "model_dir": _relative(item.model_dir, output_root),
                "checkpoint": item.checkpoint.name,
                "selection_reason": item.reason,
            }
            try:
                loaded, epoch = load_benchmark_model(
                    item.checkpoint, args.device, args.fallback_seq_len,
                    args.fallback_img_size)
                identity.update({
                    "detected_family": loaded.family,
                    "detected_version": loaded.version,
                    "checkpoint_epoch": epoch,
                    "seq_len": loaded.seq_len, "img_size": loaded.img_size,
                    "loaded_tensors": loaded.loaded_tensors,
                    "skipped_tensors": loaded.skipped_tensors,
                })
                loader = get_loader(loaded.seq_len, loaded.img_size)
                prediction, target, ids = evaluate_loaded_model(
                    loaded, loader, args.device, item.model_dir.name)
                elapsed = time.perf_counter() - started
                rows = build_metric_rows(identity, prediction, target, ids)
                for row in rows:
                    row["elapsed_seconds"] = elapsed
                write_rows(writer, file_handle, rows)
            except Exception as error:
                elapsed = time.perf_counter() - started
                write_rows(writer, file_handle,
                           [make_error_row(identity, error, elapsed)])
            finally:
                if loaded is not None:
                    del loaded.model
                    del loaded
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return csv_path


def main():
    run_benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add CLI help and mixed success/error integration tests**

Run `python scripts/evaluate_all_models.py --help` in a subprocess and assert
exit code 0. Patch discovery/adapter/loader for one successful and one failing
synthetic checkpoint; assert the CSV has three `ok` rows and one `error` row.

The mixed integration test must patch `discover_checkpoints`,
`load_benchmark_model`, `make_loader_factory`, and `evaluate_loaded_model`.
Return two `SelectedCheckpoint` values, make the second load raise
`ValueError("unsupported")`, and use zero-valued `[4,4]` prediction/target plus
IDs `[0,0,2,2]` for the first. Call `run_benchmark(args)` with a temporary CSV
path and assert row statuses equal `["ok", "ok", "ok", "error"]`.

```python
from types import SimpleNamespace
from scripts import evaluate_all_models as cli
from vla_model.benchmark import SelectedCheckpoint


class BenchmarkIntegrationTests(unittest.TestCase):
    @mock.patch.object(cli, "evaluate_loaded_model")
    @mock.patch.object(cli, "make_loader_factory")
    @mock.patch.object(cli, "load_benchmark_model")
    @mock.patch.object(cli, "discover_checkpoints")
    def test_success_and_error_are_both_persisted(
            self, discover, load_model, loader_factory, evaluate):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "output"
            first = SelectedCheckpoint(root / "V17_3", root / "V17_3/best.pt",
                                       "overall_best")
            second = SelectedCheckpoint(root / "old", root / "old/epoch_9.pt",
                                        "latest_epoch")
            discover.return_value = ([first, second], [])
            loaded = LoadedBenchmarkModel(
                mock.Mock(), "yolo", "V17.3", 8, 224, 10, 0,
                "interleaved_sincos", 1)
            load_model.side_effect = [(loaded, 12), ValueError("unsupported")]
            loader_factory.return_value = lambda *_: object()
            evaluate.return_value = (
                np.zeros((4, 4)), np.zeros((4, 4)), np.array([0, 0, 2, 2]))
            csv_path = Path(tmp) / "metrics.csv"
            args = SimpleNamespace(
                output_root=str(root), data_dir="unused", sample_ratio=.2,
                exclude_excavators=[1], batch_size=8, csv=str(csv_path),
                train_split=.857, workers=0, fallback_seq_len=8,
                fallback_img_size=224, device="cpu")
            cli.run_benchmark(args)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["status"] for row in rows],
                         ["ok", "ok", "ok", "error"])
```

- [ ] **Step 5: Run final verification**

```powershell
python -m py_compile scripts/evaluate_all_models.py vla_model/benchmark.py vla_model/visualize_yolo.py
python -m pytest tests/test_model_benchmark.py tests/test_v17_1_swing.py tests/test_v17_3_masks.py tests/test_grouped_metrics.py -q
python scripts/evaluate_all_models.py --help
git diff --check
git status --short
```

Expected: syntax and relevant tests pass; help renders; only intended files are
tracked. Existing `.claude/`, `.superpowers/`, `figure/`, `paper/`, and
`docs/v17_1_self_supervised_mask_optimization.md` remain untracked.

- [ ] **Step 6: Request review, fix findings, and commit**

Request an independent review against
`docs/superpowers/specs/2026-07-19-all-model-benchmark-design.md`. Resolve all
Critical/Important issues, rerun Step 5, then commit:

```powershell
git add -- scripts/evaluate_all_models.py vla_model/benchmark.py vla_model/visualize_yolo.py tests/test_model_benchmark.py
git commit -m "feat: add all-model CSV benchmark"
```

- [ ] **Step 7: Deliver the server command**

```bash
python scripts/evaluate_all_models.py \
  --output_root output \
  --data_dir data/excavator-motion \
  --sample_ratio 0.2 \
  --exclude_excavators 1 \
  --batch_size 8 \
  --csv output/all_model_metrics.csv
```
