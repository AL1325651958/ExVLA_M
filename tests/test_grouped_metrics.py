"""Tests for per-excavator and per-episode regression diagnostics."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.dataset import ExcavatorDataset
from vla_model.metrics import grouped_regression_metrics


class EpisodeIdDatasetTests(unittest.TestCase):
    def test_sample_exposes_its_local_episode_id(self):
        """Samples retain enough provenance for episode-level validation."""
        dataset = object.__new__(ExcavatorDataset)
        dataset.seq_len = 2
        dataset.action_chunk = 1
        dataset.img_size = 2
        dataset.split = "val"
        dataset.samples = [(1, 0)]
        frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        dataset._episodes = [
            {
                "mains_raw": frames,
                "elevations_raw": frames,
                "qpos": np.zeros((3, 4), dtype=np.float32),
                "action": np.zeros((3, 4), dtype=np.float32),
                "excavator_id": 0,
            },
            {
                "mains_raw": frames,
                "elevations_raw": frames,
                "qpos": np.zeros((3, 4), dtype=np.float32),
                "action": np.zeros((3, 4), dtype=np.float32),
                "excavator_id": 1,
            },
        ]

        sample = dataset[0]

        self.assertEqual(sample["episode_id"].dtype, torch.long)
        self.assertEqual(sample["episode_id"].item(), 1)


class GroupedRegressionMetricTests(unittest.TestCase):
    def test_reports_overall_excavator_and_episode_metrics(self):
        target = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [3.0, 1.0]])
        prediction = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])

        metrics = grouped_regression_metrics(
            prediction, target,
            excavator_ids=torch.tensor([0, 0, 1, 1]),
            episode_ids=torch.tensor([4, 4, 8, 8]),
        )

        self.assertEqual(metrics["overall"]["n_samples"], 4)
        self.assertAlmostEqual(metrics["overall"]["mae_mean"], 0.5)
        self.assertEqual(set(metrics["by_excavator"]), {0, 1})
        self.assertEqual(metrics["by_excavator"][0]["n_samples"], 2)
        self.assertEqual(metrics["by_episode"]["0:4"]["excavator_id"], 0)
        self.assertEqual(metrics["by_episode"]["0:4"]["episode_id"], 4)
        self.assertEqual(metrics["by_episode"]["1:8"]["n_samples"], 2)

    def test_constant_targets_have_finite_r2(self):
        metrics = grouped_regression_metrics(
            prediction=np.array([[1.0], [1.0]]),
            target=np.array([[1.0], [1.0]]),
            excavator_ids=[0, 0],
            episode_ids=[3, 3],
        )

        self.assertEqual(metrics["overall"]["r2"], [1.0])
        self.assertEqual(metrics["overall"]["r2_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
