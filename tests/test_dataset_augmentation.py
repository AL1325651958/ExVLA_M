"""Tests for temporally coherent clip-level image augmentation."""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.dataset import ExcavatorDataset


class ClipAugmentationTests(unittest.TestCase):
    def _dataset_with_one_clip(self):
        dataset = object.__new__(ExcavatorDataset)
        dataset.seq_len = 3
        dataset.action_chunk = 1
        dataset.img_size = 4
        dataset.split = "train"
        dataset.samples = [(0, 0)]
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        dataset._episodes = [{
            "mains_raw": frames,
            "elevations_raw": frames + 10,
            "qpos": np.zeros((4, 4), dtype=np.float32),
            "action": np.zeros((4, 4), dtype=np.float32),
            "excavator_id": 0,
        }]
        return dataset

    def test_clip_reuses_one_augmentation_for_every_timestep_and_modality(self):
        dataset = self._dataset_with_one_clip()
        sampled = {"alpha": 1.1, "beta": 3.0}
        calls = []
        dataset._sample_augmentation = lambda enabled: sampled

        def record_preprocess(image, augment=False, augmentation=None):
            calls.append(augmentation)
            return np.zeros((3, 4, 4), dtype=np.float32)

        dataset._preprocess_image = record_preprocess
        dataset[0]

        self.assertEqual(len(calls), 6)  # 3 RGB + 3 elevation frames
        self.assertTrue(all(params is sampled for params in calls))

    def test_augmentation_parameters_do_not_include_horizontal_flip(self):
        dataset = self._dataset_with_one_clip()
        params = dataset._sample_augmentation(enabled=True)
        self.assertEqual(set(params), {"alpha", "beta"})


if __name__ == "__main__":
    unittest.main()
