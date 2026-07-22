# V11 Frame-Difference Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-visual V11 motion-observation branch based on adjacent-frame residuals, correct batch-based learning-rate scheduling, and expose reliable validation diagnostics for Swing generalization.

**Architecture:** V11 computes RGB and elevation residual frames inside the model, so the HDF5 contract remains unchanged. A lightweight residual encoder produces motion tokens that are fused with the existing visual tokens before V10's temporal mask mixer. V11 inference keeps the standard three-output API and never reads qpos.

**Tech Stack:** Python, PyTorch, h5py, unittest.

---

### Task 1: Test and implement pure-visual frame residual encoding

**Files:**
- Modify: `vla_model/model_yolo.py`
- Modify: `tests/test_v10_masks.py`

- [ ] **Step 1: Write failing V11 tests**

```python
def test_v11_inference_is_invariant_to_qpos(self):
    model = self._make_model(version="v11")
    qpos_a = torch.zeros(1, 2, 4)
    qpos_b = torch.full((1, 2, 4), 99.0)
    with torch.no_grad():
        left = model(self.rgb, self.elevation, qpos_a, self.excv_id)
        right = model(self.rgb, self.elevation, qpos_b, self.excv_id)
    for a, b in zip(left, right):
        self.assertTrue(torch.equal(a, b))

def test_frame_residual_uses_only_adjacent_observations(self):
    frames = torch.tensor([[[[[1.0]]], [[[[4.0]]], [[[[9.0]]]]]])
    residual = ExcavatorVLAYolo.frame_residual(frames)
    self.assertTrue(torch.equal(residual[:, 0], torch.zeros_like(residual[:, 0])))
    self.assertEqual(residual[:, 1].item(), 3.0)
    self.assertEqual(residual[:, 2].item(), 5.0)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because V11 and `frame_residual` do not exist.

- [ ] **Step 3: Implement V11 residual fusion**

Add `version="v11"` support and a static helper:

```python
@staticmethod
def frame_residual(frames):
    return torch.cat([torch.zeros_like(frames[:, :1]), frames[:, 1:] - frames[:, :-1]], dim=1)
```

Add one shared `motion_adapter` accepting concatenated RGB/elevation residuals `[B*T, 6, H, W]`, followed by lightweight strided convolutions and adaptive pooling to the existing `G×G` grid. Project its grid features to `hidden_dim` with `motion_proj`, then fuse with vision tokens using `tokens + motion_gate(tokens) * motion_tokens`. Construct this branch only for V11. Keep `qpos` deliberately unused and return `(action, avg_masks, masks_spatial)` unless `return_aux=True` is requested by V10/V11 training.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS; V9/V10 tests remain green and V11 outputs do not depend on qpos.

- [ ] **Step 5: Commit**

```bash
git add vla_model/model_yolo.py tests/test_v10_masks.py
git commit -m "feat: add V11 frame-difference encoder"
```

### Task 2: Add a V11 training entry point with correct batch scheduler and early stopping

**Files:**
- Create: `vla_model/train_yolo_v11.py`
- Modify: `tests/test_v10_masks.py`

- [ ] **Step 1: Write failing scheduler tests**

```python
from vla_model.train_yolo_v11 import build_batch_scheduler

def test_batch_scheduler_warms_then_decays():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=3e-4)
    scheduler = build_batch_scheduler(optimizer, steps_per_epoch=10, epochs=10, warmup_ratio=0.1)
    lrs = []
    for _ in range(100):
        optimizer.step(); scheduler.step(); lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[9] > lrs[0]
    assert lrs[-1] < lrs[9]
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because `train_yolo_v11` does not exist.

- [ ] **Step 3: Implement the separate V11 trainer**

Copy `train_yolo_v10.py` to `train_yolo_v11.py`. Build `ExcavatorVLAYolo(..., version="v11")`. Implement `build_batch_scheduler` with `warmup_steps = max(1, int(steps_per_epoch * epochs * warmup_ratio))`, `LinearLR(... total_iters=warmup_steps)`, and `CosineAnnealingLR(... T_max=max(1, total_steps - warmup_steps))`. Pass scheduler into `train_epoch`, call `scheduler.step()` immediately after each optimizer update, and remove epoch-level scheduler stepping. Add `--patience` default `5`; increment a no-improvement counter after validation and break when it reaches patience. Save `model_version="v11"` and the best validation metrics.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS and the scheduler test shows warmup followed by cosine decay.

- [ ] **Step 5: Commit**

```bash
git add vla_model/train_yolo_v11.py tests/test_v10_masks.py
git commit -m "feat: add V11 training scheduler and early stopping"
```

### Task 3: Add validation metrics by excavator and episode

**Files:**
- Modify: `vla_model/dataset.py`
- Modify: `vla_model/train_yolo_v11.py`
- Modify: `tests/test_v10_masks.py`

- [ ] **Step 1: Write failing dataset metadata test**

```python
def test_dataset_sample_exposes_episode_id(self):
    sample = dataset[0]
    self.assertIn("episode_id", sample)
    self.assertEqual(sample["episode_id"].dtype, torch.long)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because samples have no `episode_id`.

- [ ] **Step 3: Implement grouped diagnostics**

Return local loaded `fidx` as `episode_id` from `ExcavatorDataset.__getitem__`; V9/V10 continue ignoring the extra key. In V11 validation, accumulate absolute errors and target/prediction statistics for each `excavator_id` and each `(excavator_id, episode_id)`. Include `by_excavator` and `by_episode` dictionaries in validation metrics, print per-joint MAE/R² for each excavator, and write the full diagnostics into `v11_history.json`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: PASS; V9/V10 tests still pass with the additional batch key.

- [ ] **Step 5: Commit**

```bash
git add vla_model/dataset.py vla_model/train_yolo_v11.py tests/test_v10_masks.py
git commit -m "feat: report V11 grouped validation metrics"
```

### Task 4: Support V11 visualization and document V12 handoff

**Files:**
- Modify: `vla_model/visualize_yolo.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-07-13-v11-v12-causal-visual-dynamics-design.md`

- [ ] **Step 1: Write a failing V11 version-detection test**

```python
def test_v11_checkpoint_keys_select_v11_model():
    state_keys = {"motion_adapter.0.weight", "temporal_mask_mixer.layers.0.self_attn.in_proj_weight"}
    self.assertEqual(detect_model_version(state_keys, {}), "v11")
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_v10_masks -v`

Expected: FAIL because `detect_model_version` does not exist.

- [ ] **Step 3: Implement compatibility and documentation**

Extract `detect_model_version(state_keys, checkpoint)` in the visualizer. Detect `motion_adapter` as V11, construct the model with `version="v11"`, and keep V9/V10 behavior unchanged. Add V11 to the changelog with the pure-visual frame-residual formulation, batch scheduler correction, early stopping, and grouped diagnostics. Append V12's offline cached optical-flow and training-only state-transition scope without implementing it in V11.

- [ ] **Step 4: Verify complete suite and V11 CLI**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

Run: `python vla_model/visualize_yolo.py --help`

Expected: exits successfully and retains `--mask_view {last,avg}`.

- [ ] **Step 5: Commit**

```bash
git add vla_model/visualize_yolo.py CHANGELOG.md docs/superpowers/specs/2026-07-13-v11-v12-causal-visual-dynamics-design.md tests/test_v10_masks.py
git commit -m "docs: add V11 visualization compatibility"
```

## Self-review

- Spec coverage: Tasks 1–3 deliver pure-visual frame residuals, corrected learning-rate units, early stopping, and Swing diagnostics; Task 4 retains visualization compatibility and documents V12 flow as a separate stage.
- Placeholder scan: no unresolved implementation placeholders remain.
- Type consistency: frame residuals preserve `[B,T,3,H,W]`; motion fusion produces `[B,T*G*G,D]`; normal V11 inference returns exactly the existing three tensors.
