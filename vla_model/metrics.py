"""Reusable regression diagnostics for excavator validation runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy arrays, CPU tensors, or GPU tensors to NumPy safely."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    absolute_error = np.abs(prediction - target)
    mae = absolute_error.mean(axis=0)
    ss_res = ((prediction - target) ** 2).sum(axis=0)
    centered = target - target.mean(axis=0, keepdims=True)
    ss_tot = (centered ** 2).sum(axis=0)

    # A constant target has no variance.  Preserve a finite, interpretable
    # score: perfect predictions get 1; any error gets 0.
    r2 = np.where(ss_res <= 1e-10, 1.0, 0.0)
    nonconstant = ss_tot > 1e-10
    r2[nonconstant] = 1.0 - ss_res[nonconstant] / ss_tot[nonconstant]
    return {
        "n_samples": int(len(target)),
        "mae": mae.tolist(),
        "mae_mean": float(mae.mean()),
        "r2": r2.tolist(),
        "r2_mean": float(r2.mean()),
    }


def grouped_regression_metrics(
    prediction: Any,
    target: Any,
    excavator_ids: Any,
    episode_ids: Any,
) -> dict:
    """Compute overall, excavator, and episode-pair MAE/R² diagnostics.

    ``episode_id`` is local to a dataset split, so episode metrics are keyed
    by ``"<excavator_id>:<episode_id>"`` to avoid collisions across machines.
    Inputs may be NumPy arrays, tensors, or lists; predictions and targets
    must have shape ``[N, joints]`` and identifier vectors must have length N.
    """
    prediction = _as_numpy(prediction)
    target = _as_numpy(target)
    excavator_ids = _as_numpy(excavator_ids).reshape(-1)
    episode_ids = _as_numpy(episode_ids).reshape(-1)

    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must have shape [N, joints]")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if len(prediction) == 0:
        raise ValueError("at least one prediction is required")
    if len(excavator_ids) != len(prediction) or len(episode_ids) != len(prediction):
        raise ValueError("identifier vectors must have one value per prediction")

    by_excavator_indices: dict[int, list[int]] = defaultdict(list)
    by_episode_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (excavator_id, episode_id) in enumerate(zip(excavator_ids, episode_ids)):
        excavator_id = int(excavator_id)
        episode_id = int(episode_id)
        by_excavator_indices[excavator_id].append(index)
        by_episode_indices[(excavator_id, episode_id)].append(index)

    by_excavator = {
        excavator_id: _regression_metrics(prediction[indices], target[indices])
        for excavator_id, indices in sorted(by_excavator_indices.items())
    }
    by_episode = {}
    for (excavator_id, episode_id), indices in sorted(by_episode_indices.items()):
        metric = _regression_metrics(prediction[indices], target[indices])
        metric.update(excavator_id=excavator_id, episode_id=episode_id)
        by_episode[f"{excavator_id}:{episode_id}"] = metric

    return {
        "overall": _regression_metrics(prediction, target),
        "by_excavator": by_excavator,
        "by_episode": by_episode,
    }
