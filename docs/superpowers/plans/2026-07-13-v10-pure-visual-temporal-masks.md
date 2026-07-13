# V10 Pure-Visual Temporal Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add V10 temporal, vision-only joint masks with training-only qpos supervision while preserving V9 checkpoint visualization and the three-value inference API.

**Architecture:** `ExcavatorVLAYolo` gains an optional V10 temporal mask mixer operating across frames at each grid location. The normal `forward` still returns `(action, avg_masks, masks_spatial)` and ignores qpos; only `return_aux=True`, used by V10 training, additionally returns a pose auxiliary prediction. The visualizer detects V10 weights and selects raw last-frame masks by default, but still supports V9 and average masks.

**Tech Stack:** Python, PyTorch, unittest, h5py, OpenCV.

---

### Task 1: Add test coverage for the pure-visual V10 interface

**Files:**
- Create: `tests/test_v10_masks.py`
- Modify: `vla_model/model_yolo.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import torch
from vla_model.model_yolo import ExcavatorVLAYolo


class V10MaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ExcavatorVLAYolo(
            seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
            n_layers=1, ff_dim=64, dropout=0.0, version="v10",
        ).eval()
        self.rgb = torch.randn(1, 2, 3, 32, 32)
        self.elevation = torch.randn(1, 2, 3, 32, 32)
        self.excv_id = torch.zeros(1, dtype=torch.long)

    def test_v10_inference_is_invariant_to_qpos(self):
        qpos_a = torch.zeros(1, 2, 4)
        qpos_b = torch.ones(1, 2, 4) * 99
        with torch.no_grad():
            out_a = self.model(self.rgb, self.elevation, qpos_a, self.excv_id)
            out_b = self.model(self.rgb, self.elevation, qpos_b, self.excv_id)
        for left, right in zip(out_a, out_b):
            self.assertTrue(torch.equal(left, right))

    def test_v10_training_auxiliary_output_is_opt_in(self):
        outputs = self.model(self.rgb, self.elevation, None, self.excv_id, return_aux=True)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[3].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because `ExcavatorVLAYolo` has no `version` or `return_aux` arguments.

- [ ] **Step 3: Implement the V10 model interface**

Add a `TemporalMaskMixer` in `vla_model/model_yolo.py` that reshapes `[B, T, G, G, D]` to `[B*G*G, T, D]`, applies a `batch_first=True` Transformer encoder layer, and restores the original shape. Add `version="v9"` to the model constructor; construct the mixer and `pose_aux_head = nn.Linear(hidden_dim, 4)` only when `version == "v10"`. In `forward`, run the mixer before mask generation for V10, do not read `qpos`, and append `pose_aux_head(memory[:, -G*G:].mean(1))` only when `return_aux=True`.

```python
def forward(self, rgb, elevation, qpos=None, excavator_id=None, return_aux=False):
    # qpos is deliberately unused: inference remains vision-only.
    ...
    mask_tokens = self.temporal_mask_mixer(grid) if self.version == "v10" else grid
    tokens = mask_tokens.reshape(B, T * G * G, D)
    ...
    outputs = (action, avg_masks, masks_spatial)
    if return_aux:
        return (*outputs, self.pose_aux_head(memory[:, -G * G:].mean(dim=1)))
    return outputs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS; qpos changes do not change any normal inference output and auxiliary output is `[B, 4]` only when requested.

- [ ] **Step 5: Commit**

```bash
git add tests/test_v10_masks.py vla_model/model_yolo.py
git commit -m "feat: add V10 pure-visual temporal masks"
```

### Task 2: Add training-only qpos auxiliary supervision and V10 entry point

**Files:**
- Create: `vla_model/train_yolo_v10.py`
- Modify: `vla_model/config.py`
- Test: `tests/test_v10_masks.py`

- [ ] **Step 1: Extend the failing test with V10 checkpoint metadata**

```python
def test_v10_config_selects_the_v10_model(self):
    from vla_model.train_yolo_v10 import build_v10_model
    model = build_v10_model(seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
                            n_layers=1, ff_dim=64, dropout=0.0)
    self.assertEqual(model.version, "v10")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_v10_masks.V10MaskTests.test_v10_config_selects_the_v10_model -v`

Expected: FAIL because `train_yolo_v10` and `build_v10_model` do not exist.

- [ ] **Step 3: Implement V10 training as a separate entry point**

Copy `vla_model/train_yolo.py` to `vla_model/train_yolo_v10.py`. Add `Config.v10_pose_aux_weight: float = 0.05`. Implement `build_v10_model(...)` returning `ExcavatorVLAYolo(..., version="v10")`. In `train_epoch`, invoke the model with `return_aux=True`, compute a training-only `MSELoss` against `qpos[:, -1]`, and add `config.v10_pose_aux_weight * pose_aux_loss` to the existing loss. In `validate`, call the normal three-output forward path and do not calculate the auxiliary loss. Save `"model_version": "v10"` in V10 checkpoints.

```python
raw_out, masks_avg, masks_spatial, pose_aux = model(
    rgb, elevation, qpos, excavator_id, return_aux=True
)
pose_aux_loss = criterion(pose_aux, qpos[:, -1])
loss = pred_loss + circle_loss + area_loss + diversity_loss + temp_loss \
       + config.v10_pose_aux_weight * pose_aux_loss
```

- [ ] **Step 4: Run all V10 unit tests**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vla_model/train_yolo_v10.py vla_model/config.py tests/test_v10_masks.py
git commit -m "feat: train V10 with qpos auxiliary supervision"
```

### Task 3: Make visualization select raw V10 masks while retaining V9 compatibility

**Files:**
- Modify: `vla_model/visualize_yolo.py`
- Modify: `tests/test_v10_masks.py`

- [ ] **Step 1: Write failing mask-view selection tests**

```python
from vla_model.visualize_yolo import select_mask_view

def test_last_mask_view_selects_last_raw_frame(self):
    masks = torch.arange(1 * 4 * 2 * 3 * 3).view(1, 4, 2, 3, 3)
    self.assertTrue(torch.equal(select_mask_view(masks, "last"), masks[:, :, -1]))

def test_average_mask_view_preserves_v9_behavior(self):
    masks = torch.arange(1 * 4 * 2 * 3 * 3).view(1, 4, 2, 3, 3)
    self.assertTrue(torch.equal(select_mask_view(masks, "avg"), masks.mean(dim=2)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because `select_mask_view` does not exist.

- [ ] **Step 3: Implement backward-compatible visualization**

Add `select_mask_view(masks_spatial, mask_view)` returning either `masks_spatial[:, :, -1]` or `masks_spatial.mean(dim=2)`, raising `ValueError` for another value. Add `--mask_view {last,avg}` to the CLI. Detect V10 by checkpoint metadata or a `temporal_mask_mixer` state key; use `last` as its default, otherwise use `avg`. Build V10 models with `version="v10"`, retain the V9 model construction path, and always read `outputs[2]` for raw masks when available.

```python
def select_mask_view(masks_spatial, mask_view):
    if mask_view == "last":
        return masks_spatial[:, :, -1]
    if mask_view == "avg":
        return masks_spatial.mean(dim=2)
    raise ValueError(f"Unknown mask view: {mask_view}")
```

- [ ] **Step 4: Run tests and smoke-test both checkpoint formats**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS.

Run: `python vla_model/visualize_yolo.py --help`

Expected: usage includes `--mask_view {last,avg}`.

- [ ] **Step 5: Commit**

```bash
git add vla_model/visualize_yolo.py tests/test_v10_masks.py
git commit -m "feat: visualize raw V10 temporal masks"
```

### Task 4: Verify compatibility and document V10 invocation

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a V10 changelog entry**

Document pure-visual inference, training-only qpos auxiliary loss, V10 checkpoint metadata, and the `--mask_view` modes. State explicitly that V9 checkpoints still load and default to average masks.

- [ ] **Step 2: Run model smoke tests**

Run: `python -c "import torch; from vla_model.model_yolo import ExcavatorVLAYolo; m=ExcavatorVLAYolo(seq_len=2,img_size=32,hidden_dim=32,n_heads=4,n_layers=1,ff_dim=64,version='v10').eval(); x=torch.randn(1,2,3,32,32); e=torch.randn_like(x); i=torch.zeros(1,dtype=torch.long); print([z.shape for z in m(x,e,None,i)])"`

Expected: three shapes `[1, 8]`, `[1, 4, 2, 2]`, and `[1, 4, 2, 2, 2]`; no qpos input is required.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: describe V10 pure-visual workflow"
```

## Self-review

- Spec coverage: Tasks 1 and 2 provide temporal visual masks and qpos-only training supervision; Task 3 provides raw-mask visualization and V9 fallback; Task 4 verifies and documents compatibility.
- Placeholder scan: no TBD/TODO items remain.
- Type consistency: normal inference always returns three tensors; `return_aux=True` adds a fourth `[B,4]` pose tensor only for V10 training.
