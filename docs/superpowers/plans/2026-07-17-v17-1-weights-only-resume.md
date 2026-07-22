# V17.1 Weights-Only Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe weights-only warm-start mode to V17.1 training.

**Architecture:** A pure helper selects checkpoint model state and training metadata for either full resume or weights-only warm start. The CLI uses this helper before compatible state loading and conditionally restores optimizer/scheduler state.

**Tech Stack:** Python, PyTorch, argparse, unittest

---

### Task 1: Specify Resume Selection

**Files:**
- Modify: `tests/test_v17_1_swing.py`
- Modify: `vla_model/train_yolo_v17_1.py`

- [x] **Step 1: Write failing tests** for EMA selection/reset behavior in weights-only mode and unchanged full-resume behavior.
- [x] **Step 2: Run the focused tests** and confirm failure because `prepare_resume` is missing.
- [x] **Step 3: Implement `prepare_resume`** to return model state, EMA state, epoch, best metrics, and whether training state should be restored.
- [x] **Step 4: Run the focused tests** and confirm both resume modes pass.

### Task 2: Expose the CLI

**Files:**
- Modify: `vla_model/train_yolo_v17_1.py`
- Test: `tests/test_v17_1_swing.py`

- [x] **Step 1: Add `--weights_only`** and reject it when `--resume` is absent.
- [x] **Step 2: Route checkpoint loading through `prepare_resume`** and skip `restore_training_state` in weights-only mode.
- [x] **Step 3: Print an explicit weights-only startup message** including epoch reset and fresh optimizer/scheduler behavior.
- [x] **Step 4: Run V17.1 and grouped-metrics tests**, compile changed scripts, and run `git diff --check`.
- [x] **Step 5: Commit and push `main`** after verifying only intended files are staged.
