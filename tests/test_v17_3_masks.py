"""Focused tests for the V17.3 mask-regularization training variant."""

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
        with self.assertRaisesRegex(ValueError, r"\[B, 2, 4, T, G, G\]"):
            compute_mask_diversity_loss(
                torch.zeros(1, 4, 2, 2), mode="within_modality", margin=0.5
            )


if __name__ == "__main__":
    unittest.main()
