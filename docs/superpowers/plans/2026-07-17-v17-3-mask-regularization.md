# STVTA V17.3 Mask Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent V17.3 trainer that fixes cross-modal mask repulsion, records mask-location diagnostics, preserves V17.1 behavior, and remains state-dict compatible with V17.1.

**Architecture:** Keep `ExcavatorVLAYolo(version="v17.1")` as the shared network topology and make V17.3 a training-objective variant. Pure helper functions own diversity loss and diagnostics; a thin V17.3 entry point selects variant metadata; the visualizer maps V17.3 metadata back to the V17.1-compatible topology.

**Tech Stack:** Python 3.10, PyTorch, NumPy, unittest/pytest, existing STVTA training and visualization modules.

---

## File Map

- Modify `vla_model/train_yolo_v17_1.py`: variant configuration, pure mask helpers, validation aggregation, versioned artifact names.
- Create `vla_model/train_yolo_v17_3.py`: thin V17.3 executable entry point.
- Modify `vla_model/visualize_yolo.py`: V17.3 checkpoint metadata detection.
- Create `tests/test_v17_3_masks.py`: focused loss, diagnostic, artifact, and compatibility tests.
- Re-run `tests/test_v17_1_swing.py`: prove V17.1 defaults remain unchanged.

### Task 1: Specify V17.3 mask behavior with failing tests

**Files:**
- Create: `tests/test_v17_3_masks.py`
- Read: `vla_model/train_yolo_v17_1.py`
- Read: `vla_model/visualize_yolo.py`

- [ ] **Step 1: Write failing tests for variant selection and diversity-pair semantics**

```python
import unittest

import torch

from vla_model.train_yolo_v17_1 import (
    compute_mask_diversity_loss,
    get_training_variant,
)


class V173DiversityTests(unittest.TestCase):
    @staticmethod
    def one_hot_joint_masks():
        masks = torch.zeros(1, 2, 4, 1, 2, 2)
        for joint in range(4):
            y, x = divmod(joint, 2)
            masks[:, :, joint, :, y, x] = 1.0
        return masks

    def test_v17_1_defaults_are_legacy(self):
        variant = get_training_variant("v17.1")
        self.assertEqual(variant["diversity_mode"], "legacy_all_pairs")
        self.assertEqual(variant["diversity_margin"], 0.3)
        self.assertFalse(variant["mask_diagnostics"])

    def test_v17_3_uses_within_modality_pairs(self):
        variant = get_training_variant("v17.3")
        self.assertEqual(variant["diversity_mode"], "within_modality")
        self.assertEqual(variant["diversity_margin"], 0.5)
        self.assertTrue(variant["mask_diagnostics"])

    def test_same_joint_cross_modal_agreement_is_not_penalized(self):
        loss = compute_mask_diversity_loss(
            self.one_hot_joint_masks(), mode="within_modality", margin=0.5
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_same_modality_joint_collapse_is_penalized(self):
        masks = self.one_hot_joint_masks()
        masks[:, :, 1] = masks[:, :, 0]
        loss = compute_mask_diversity_loss(
            masks, mode="within_modality", margin=0.5
        )
        self.assertGreater(loss.item(), 0.0)

    def test_invalid_mask_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "\[B, 2, 4, T, G, G\]"):
            compute_mask_diversity_loss(
                torch.zeros(1, 4, 2, 2), mode="within_modality", margin=0.5
            )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: collection fails because `compute_mask_diversity_loss` and
`get_training_variant` do not exist.

- [ ] **Step 3: Add failing tests for mask diagnostics**

```python
from vla_model.train_yolo_v17_1 import compute_mask_diagnostics


class V173DiagnosticTests(unittest.TestCase):
    def test_diagnostics_report_area_centroid_and_cross_modal_similarity(self):
        masks = torch.zeros(1, 2, 4, 1, 2, 2)
        masks[:, :, 0, :, 0, 0] = 1.0
        masks[:, :, 2, :, 1, 1] = 1.0
        diagnostics = compute_mask_diagnostics(masks)

        self.assertEqual(tuple(diagnostics["area"].shape), (1, 2, 4))
        self.assertAlmostEqual(diagnostics["area"][0, 0, 0].item(), 0.25)
        self.assertAlmostEqual(diagnostics["center_x"][0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(diagnostics["center_y"][0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(diagnostics["center_x"][0, 0, 2].item(), 1.0)
        self.assertAlmostEqual(diagnostics["center_y"][0, 0, 2].item(), 1.0)
        self.assertAlmostEqual(
            diagnostics["cross_modal_similarity"][0, 0].item(), 1.0
        )
```

- [ ] **Step 4: Re-run and verify the expected missing-helper failure**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: FAIL during import for the new diagnostic helper.

- [ ] **Step 5: Commit the RED tests**

```powershell
git add -- tests/test_v17_3_masks.py
git commit -m "test: define V17.3 mask regularization"
```

### Task 2: Implement pure variant, diversity, and diagnostic helpers

**Files:**
- Modify: `vla_model/train_yolo_v17_1.py`
- Test: `tests/test_v17_3_masks.py`

- [ ] **Step 1: Add immutable-by-copy training variant lookup**

Add near the shared utilities:

```python
_TRAINING_VARIANTS = {
    "v17.1": {
        "model_version": "v17.1",
        "checkpoint_prefix": "yolo_v17_1",
        "diversity_mode": "legacy_all_pairs",
        "diversity_margin": 0.3,
        "mask_diagnostics": False,
    },
    "v17.3": {
        "model_version": "v17.3",
        "checkpoint_prefix": "yolo_v17_3",
        "diversity_mode": "within_modality",
        "diversity_margin": 0.5,
        "mask_diagnostics": True,
    },
}


def get_training_variant(version):
    try:
        return dict(_TRAINING_VARIANTS[str(version).lower()])
    except KeyError as error:
        supported = ", ".join(sorted(_TRAINING_VARIANTS))
        raise ValueError(
            f"unknown training variant {version!r}; expected one of: {supported}"
        ) from error
```

- [ ] **Step 2: Add the shape guard and diversity helper**

```python
def _validate_dual_mask_shape(masks_spatial):
    if (
        masks_spatial.ndim != 6
        or masks_spatial.shape[1] != 2
        or masks_spatial.shape[2] != 4
    ):
        raise ValueError(
            "masks_spatial must have shape [B, 2, 4, T, G, G]"
        )


def compute_mask_diversity_loss(masks_spatial, mode, margin):
    _validate_dual_mask_shape(masks_spatial)
    if mode == "legacy_all_pairs":
        batch = masks_spatial.size(0)
        flat = masks_spatial.reshape(batch, 8, -1)
        normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        eye = torch.eye(8, device=similarity.device, dtype=similarity.dtype)
        off_diagonal = similarity * (1.0 - eye.unsqueeze(0))
        return 0.5 * torch.relu(off_diagonal - margin).pow(2).mean()
    if mode == "within_modality":
        flat = masks_spatial.flatten(start_dim=3)
        normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal = ~torch.eye(4, device=similarity.device, dtype=torch.bool)
        penalties = torch.relu(similarity - margin).pow(2)
        return 0.5 * penalties[..., off_diagonal].mean()
    raise ValueError(f"unknown mask diversity mode: {mode!r}")
```

The legacy branch deliberately keeps the original diagonal-zeroing and mean
denominator so V17.1 numerical behavior remains unchanged.

- [ ] **Step 3: Add detached per-sample mask diagnostics**

```python
@torch.no_grad()
def compute_mask_diagnostics(masks_spatial):
    _validate_dual_mask_shape(masks_spatial)
    masks = masks_spatial.detach()
    batch, _, _, _, grid_h, grid_w = masks.shape
    area = masks.mean(dim=(-3, -2, -1))
    mass = masks.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    x_axis = torch.linspace(0.0, 1.0, grid_w, device=masks.device, dtype=masks.dtype)
    y_axis = torch.linspace(0.0, 1.0, grid_h, device=masks.device, dtype=masks.dtype)
    center_x = (masks * x_axis.view(1, 1, 1, 1, 1, grid_w)).sum(
        dim=(-3, -2, -1)
    ) / mass
    center_y = (masks * y_axis.view(1, 1, 1, 1, grid_h, 1)).sum(
        dim=(-3, -2, -1)
    ) / mass
    flat = masks.reshape(batch, 2, 4, -1)
    cross_modal_similarity = torch.nn.functional.cosine_similarity(
        flat[:, 0], flat[:, 1], dim=-1, eps=1e-6
    )
    return {
        "area": area,
        "center_x": center_x,
        "center_y": center_y,
        "cross_modal_similarity": cross_modal_similarity,
    }
```

- [ ] **Step 4: Replace the inline V17.1 diversity calculation with the helper**

```python
diversity_loss = compute_mask_diversity_loss(
    masks_spatial,
    mode=getattr(config, "v17_mask_diversity_mode", "legacy_all_pairs"),
    margin=getattr(config, "v17_mask_diversity_margin", 0.3),
)
```

- [ ] **Step 5: Run focused and V17.1 regression tests**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py tests/test_v17_1_swing.py -q
```

Expected: diversity and diagnostic tests PASS; existing V17.1 tests PASS.

- [ ] **Step 6: Commit the helper implementation**

```powershell
git add -- vla_model/train_yolo_v17_1.py tests/test_v17_3_masks.py
git commit -m "fix: scope V17.3 mask diversity by modality"
```

### Task 3: Add V17.3 validation diagnostics

**Files:**
- Modify: `vla_model/train_yolo_v17_1.py`
- Modify: `tests/test_v17_3_masks.py`

- [ ] **Step 1: Write a failing aggregation and formatting test**

Add these imports and methods inside `V173DiagnosticTests`:

```python
from vla_model.train_yolo_v17_1 import (
    accumulate_mask_diagnostics,
    format_mask_diagnostics,
)


def test_diagnostic_accumulator_is_sample_weighted(self):
    totals = None
    first = {
        "area": torch.ones(2, 2, 4),
        "center_x": torch.zeros(2, 2, 4),
        "center_y": torch.full((2, 2, 4), 0.25),
        "cross_modal_similarity": torch.full((2, 4), 0.5),
    }
    totals, count = accumulate_mask_diagnostics(totals, 0, first)
    second = {key: value[:1] * 3 for key, value in first.items()}
    totals, count = accumulate_mask_diagnostics(totals, count, second)
    averaged = {key: (value / count).tolist() for key, value in totals.items()}
    self.assertEqual(count, 3)
    self.assertAlmostEqual(averaged["area"][0][0], 5.0 / 3.0)

def test_formatter_emphasizes_boom_and_bucket(self):
    diagnostics = {
        "area": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        "center_x": [[0.5] * 4, [0.6] * 4],
        "center_y": [[0.1, 0.2, 0.8, 0.4], [0.2, 0.3, 0.7, 0.5]],
        "cross_modal_similarity": [0.9, 0.8, 0.7, 0.6],
    }
    lines = format_mask_diagnostics(diagnostics)
    self.assertEqual(len(lines), 2)
    self.assertIn("Boom", lines[0])
    self.assertIn("Bucket", lines[1])
    self.assertNotIn("Swing", " ".join(lines))
```

- [ ] **Step 2: Run the accumulator test and verify RED**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: FAIL because `accumulate_mask_diagnostics` does not exist.

- [ ] **Step 3: Implement sample-weighted accumulation and validation output**

```python
def accumulate_mask_diagnostics(totals, count, diagnostics):
    batch_size = diagnostics["area"].shape[0]
    batch_totals = {key: value.detach().sum(dim=0).cpu() for key, value in diagnostics.items()}
    if totals is None:
        totals = batch_totals
    else:
        for key in totals:
            totals[key] += batch_totals[key]
    return totals, count + batch_size
```

At the beginning of `validate`, initialize:

```python
diagnostic_totals = None
diagnostic_count = 0
```

Retain the masks returned by the model and aggregate only for V17.3:

```python
raw_out, _, masks_spatial = model(rgb, elevation, qpos, excavator_id)
if getattr(config, "v17_mask_diagnostics", False):
    diagnostic_totals, diagnostic_count = accumulate_mask_diagnostics(
        diagnostic_totals,
        diagnostic_count,
        compute_mask_diagnostics(masks_spatial),
    )
```

Build the existing return dictionary as `metrics`, then append diagnostics:

```python
if diagnostic_count:
    metrics["mask_diagnostics"] = {
        key: (value / diagnostic_count).tolist()
        for key, value in diagnostic_totals.items()
    }
return metrics
```

- [ ] **Step 4: Print and persist Boom/Bucket diagnostics per epoch**

Add the formatter:

```python
def format_mask_diagnostics(diagnostics):
    if not diagnostics:
        return []
    lines = []
    modalities = ("RGB", "Elevation")
    joints = ((0, "Boom"), (2, "Bucket"))
    for joint_index, joint_name in joints:
        modality_parts = []
        for modality_index, modality_name in enumerate(modalities):
            area = diagnostics["area"][modality_index][joint_index]
            center_x = diagnostics["center_x"][modality_index][joint_index]
            center_y = diagnostics["center_y"][modality_index][joint_index]
            modality_parts.append(
                f"{modality_name}: area={area:.3f}, center=({center_x:.3f},{center_y:.3f})"
            )
        similarity = diagnostics["cross_modal_similarity"][joint_index]
        lines.append(
            f"  Mask {joint_name} | " + " | ".join(modality_parts)
            + f" | RGB/Elev cos={similarity:.3f}"
        )
    return lines
```

Initialize history conditionally and append/print after each validation:

```python
if config.v17_mask_diagnostics:
    history["val_mask_diagnostics"] = []

# Inside the epoch loop, after validation:
if config.v17_mask_diagnostics:
    diagnostics = val_metrics.get("mask_diagnostics", {})
    history["val_mask_diagnostics"].append(diagnostics)
    for line in format_mask_diagnostics(diagnostics):
        print(line)
```

Do not add any diagnostic value to the training loss.

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit diagnostics**

```powershell
git add -- vla_model/train_yolo_v17_1.py tests/test_v17_3_masks.py
git commit -m "feat: log V17.3 mask diagnostics"
```

### Task 4: Add the independent V17.3 entry point and artifacts

**Files:**
- Create: `vla_model/train_yolo_v17_3.py`
- Modify: `vla_model/train_yolo_v17_1.py`
- Modify: `tests/test_v17_3_masks.py`

- [ ] **Step 1: Write failing tests for checkpoint metadata and entry-point selection**

Add imports:

```python
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from torch import nn

from vla_model.train_yolo_v17_1 import save_checkpoint
```

Add these tests:

```python
class V173ArtifactTests(unittest.TestCase):
    def test_checkpoint_uses_v17_3_filename_and_metadata(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(
                output_dir=tmp,
                v17_checkpoint_prefix="yolo_v17_3",
                v17_model_version="v17.3",
            )
            path = save_checkpoint(
                model,
                optimizer,
                scaler,
                scheduler,
                epoch=0,
                metrics={"loss": 1.0, "r2": [0.0] * 4},
                config=config,
                is_best=True,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(Path(path).name, "yolo_v17_3_checkpoint_best.pt")
            self.assertEqual(checkpoint["model_version"], "v17.3")

    def test_entry_point_selects_v17_3(self):
        import vla_model.train_yolo_v17_3 as entry_point

        with mock.patch.object(
            entry_point, "train_v17_1_main", return_value=17
        ) as run:
            self.assertEqual(entry_point.main(), 17)
        run.assert_called_once_with(training_version="v17.3")
```

- [ ] **Step 2: Run the artifact tests and verify RED**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: FAIL because checkpoint naming is hard-coded to V17.1 and the V17.3
module is absent.

- [ ] **Step 3: Parameterize main and artifact metadata without changing defaults**

Change the entry signature to:

```python
def main(training_version="v17.1"):
    variant = get_training_variant(training_version)
```

After creating `Config`, assign:

```python
config.v17_model_version = variant["model_version"]
config.v17_checkpoint_prefix = variant["checkpoint_prefix"]
config.v17_mask_diversity_mode = variant["diversity_mode"]
config.v17_mask_diversity_margin = variant["diversity_margin"]
config.v17_mask_diagnostics = variant["mask_diagnostics"]
```

Use these values in console labels, checkpoint filename, checkpoint
`model_version`, and history filename. Every `getattr` fallback remains the old
V17.1 value.

Replace hard-coded values in `save_checkpoint`:

```python
checkpoint_prefix = getattr(config, "v17_checkpoint_prefix", "yolo_v17_1")
model_version = getattr(config, "v17_model_version", "v17.1")
path = os.path.join(
    config.output_dir, f"{checkpoint_prefix}_checkpoint_{suffix}.pt"
)
# In payload:
"model_version": model_version,
```

Replace the history filename with:

```python
history_name = f"{config.v17_checkpoint_prefix}_history.json"
with open(os.path.join(config.output_dir, history_name), "w") as file:
    json.dump(history, file, indent=2)
```

- [ ] **Step 4: Create the thin V17.3 executable**

```python
"""Train STVTA V17.3 with within-modality mask diversity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.train_yolo_v17_1 import main as train_v17_1_main


def main():
    return train_v17_1_main(training_version="v17.3")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run artifact and V17.1 regression tests**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py tests/test_v17_1_swing.py -q
```

Expected: PASS, including legacy V17.1 checkpoint naming tests.

- [ ] **Step 6: Commit the V17.3 entry point**

```powershell
git add -- vla_model/train_yolo_v17_1.py vla_model/train_yolo_v17_3.py tests/test_v17_3_masks.py
git commit -m "feat: add STVTA V17.3 training entry point"
```

### Task 5: Make visualization metadata-compatible

**Files:**
- Modify: `vla_model/visualize_yolo.py`
- Modify: `tests/test_v17_3_masks.py`

- [ ] **Step 1: Add a failing V17.3 detection test**

```python
from vla_model.visualize_yolo import detect_version


def test_visualizer_maps_v17_3_to_v17_1_topology(self):
    result = detect_version(
        {"joint_logit_bias", "temporal_mask_mixer.layers.0.weight"},
        checkpoint_version="v17.3",
    )
    self.assertEqual(result, ("V17.3", True, "v17.1"))
```

- [ ] **Step 2: Run the detection test and verify RED**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py -q
```

Expected: FAIL because the metadata currently falls through to V17.1.

- [ ] **Step 3: Add explicit V17.3 metadata handling before V17.1 detection**

```python
checkpoint_version = str(checkpoint_version).lower()
if checkpoint_version == "v17.3":
    return "V17.3", True, "v17.1"
if checkpoint_version == "v17.1":
    return "V17.1", True, "v17.1"
```

The returned architecture version remains `v17.1` because V17.3 introduces no
new parameters or forward-path modules.

- [ ] **Step 4: Run focused visualization and version tests**

Run:

```powershell
python -m pytest tests/test_v17_3_masks.py tests/test_v17_1_swing.py tests/test_v10_masks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit visualization compatibility**

```powershell
git add -- vla_model/visualize_yolo.py tests/test_v17_3_masks.py
git commit -m "fix: recognize V17.3 checkpoints in visualizer"
```

### Task 6: Final verification and training handoff

**Files:**
- Verify: `vla_model/train_yolo_v17_1.py`
- Verify: `vla_model/train_yolo_v17_3.py`
- Verify: `vla_model/visualize_yolo.py`
- Verify: `tests/test_v17_3_masks.py`

- [ ] **Step 1: Run syntax compilation**

```powershell
python -m py_compile vla_model/train_yolo_v17_1.py vla_model/train_yolo_v17_3.py vla_model/visualize_yolo.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run all focused regression tests**

```powershell
python -m pytest tests/test_v17_3_masks.py tests/test_v17_1_swing.py tests/test_v10_masks.py tests/test_train_yolo_v11.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Inspect the final diff and artifact scope**

```powershell
git diff --check
git status --short
git diff --stat HEAD~5..HEAD
```

Expected: no whitespace errors; `.claude/`, `.superpowers/`, `figure/`, and
`docs/v17_1_self_supervised_mask_optimization.md` remain untracked and unstaged.

- [ ] **Step 4: Run a command-line import smoke test**

```powershell
python -c "from vla_model.train_yolo_v17_1 import get_training_variant; print(get_training_variant('v17.3'))"
```

Expected: a dictionary containing `within_modality`, margin `0.5`, and V17.3
artifact metadata.

- [ ] **Step 5: Provide the server training command**

Fresh training:

```bash
python vla_model/train_yolo_v17_3.py \
  --data_dir data/excavator-motion \
  --epochs 80 \
  --batch_size 12 \
  --img_size 224 \
  --seq_len 8 \
  --sample_ratio 0.1 \
  --output_dir output/V17_3
```

Recommended V17.1 warm start:

```bash
python vla_model/train_yolo_v17_3.py \
  --data_dir data/excavator-motion \
  --epochs 80 \
  --batch_size 12 \
  --img_size 224 \
  --seq_len 8 \
  --sample_ratio 0.1 \
  --output_dir output/V17_3 \
  --resume output/V17_1_swing_fix_restart/yolo_v17_1_checkpoint_best.pt \
  --weights_only
```

- [ ] **Step 6: Request code review, address findings, and make a final commit if needed**

Stage only the V17.3 implementation, tests, and intended docs. Do not stage any
pre-existing user files or visual-companion artifacts.
