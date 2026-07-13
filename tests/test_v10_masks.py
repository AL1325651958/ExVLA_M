"""Tests for V10 pure-visual temporal masks interface."""
import unittest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model_yolo import ExcavatorVLAYolo


class V10MaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.v9 = ExcavatorVLAYolo(
            seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
            n_layers=1, ff_dim=64, dropout=0.0,
        ).eval()
        self.v10 = ExcavatorVLAYolo(
            seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
            n_layers=1, ff_dim=64, dropout=0.0, version="v10",
        ).eval()
        self.rgb = torch.randn(1, 2, 3, 32, 32)
        self.elevation = torch.randn(1, 2, 3, 32, 32)
        self.excv_id = torch.zeros(1, dtype=torch.long)

    def test_v9_exists_and_runs(self):
        """V9 model still works unchanged."""
        with torch.no_grad():
            out = self.v9(self.rgb, self.elevation, None, self.excv_id)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (1, 8))
        self.assertEqual(out[1].shape, (1, 4, 2, 2))
        self.assertEqual(out[2].shape, (1, 4, 2, 2, 2))

    def test_v10_inference_is_invariant_to_qpos(self):
        """qpos changes must not alter any V10 inference output."""
        qpos_a = torch.zeros(1, 2, 4)
        qpos_b = torch.ones(1, 2, 4) * 99
        with torch.no_grad():
            out_a = self.v10(self.rgb, self.elevation, qpos_a, self.excv_id)
            out_b = self.v10(self.rgb, self.elevation, qpos_b, self.excv_id)
        for left, right in zip(out_a, out_b):
            self.assertTrue(torch.equal(left, right),
                            f"qpos changed V10 output: max diff "
                            f"{max((l - r).abs().max().item() for l, r in zip(out_a, out_b)):.6f}")

    def test_v10_training_auxiliary_output_is_opt_in(self):
        """Normal inference returns 3 tensors; return_aux=True returns 4."""
        out3 = self.v10(self.rgb, self.elevation, None, self.excv_id)
        self.assertEqual(len(out3), 3)

        out4 = self.v10(self.rgb, self.elevation, None, self.excv_id, return_aux=True)
        self.assertEqual(len(out4), 4)
        self.assertEqual(out4[3].shape, (1, 4))  # pose_aux: [B, 4]

    def test_v10_pose_aux_is_qpos_prediction(self):
        """Auxiliary output is a 4-DOF pose prediction in radians."""
        self.v10.train()
        out4 = self.v10(self.rgb, self.elevation, None, self.excv_id, return_aux=True)
        self.assertEqual(out4[3].shape, (1, 4))

    def test_v10_shape_matches_v9_for_inference(self):
        """Normal 3-output shapes identical between V9 and V10."""
        with torch.no_grad():
            v9 = self.v9(self.rgb, self.elevation, None, self.excv_id)
            v10 = self.v10(self.rgb, self.elevation, None, self.excv_id)
        self.assertEqual(v9[0].shape, v10[0].shape)
        self.assertEqual(v9[1].shape, v10[1].shape)
        self.assertEqual(v9[2].shape, v10[2].shape)

    def test_v10_has_temporal_mask_mixer(self):
        self.assertTrue(hasattr(self.v10, "temporal_mask_mixer"))
        self.assertEqual(self.v10.version, "v10")

    def test_v9_has_no_temporal_mixer(self):
        self.assertTrue(self.v9.temporal_mask_mixer is None)


class V11FrameResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.v11 = ExcavatorVLAYolo(
            seq_len=3, img_size=32, hidden_dim=32, n_heads=4,
            n_layers=1, ff_dim=64, dropout=0.0, version="v11",
        ).eval()
        self.rgb = torch.randn(1, 3, 3, 32, 32)
        self.elevation = torch.randn(1, 3, 3, 32, 32)
        self.excv_id = torch.zeros(1, dtype=torch.long)

    def test_v11_inference_is_invariant_to_qpos(self):
        """V11 remains a pure-visual model when qpos values change."""
        qpos_a = torch.zeros(1, 3, 4)
        qpos_b = torch.full((1, 3, 4), -17.0)
        with torch.no_grad():
            out_a = self.v11(self.rgb, self.elevation, qpos_a, self.excv_id)
            out_b = self.v11(self.rgb, self.elevation, qpos_b, self.excv_id)
        for left, right in zip(out_a, out_b):
            self.assertTrue(torch.equal(left, right))

    def test_v11_frame_residual_zeros_first_timestep_and_differences_following(self):
        """Motion input is the adjacent-frame residual, with a zero first frame."""
        frames = torch.tensor([[[[[1.0]]], [[[4.0]]], [[[10.0]]]]])
        residual = self.v11.frame_residual(frames)
        expected = torch.tensor([[[[[0.0]]], [[[3.0]]], [[[6.0]]]]])
        self.assertTrue(torch.equal(residual, expected))

    def test_v11_has_motion_adapter(self):
        self.assertEqual(self.v11.version, "v11")
        self.assertIsNotNone(self.v11.motion_adapter)


class V10TrainingTests(unittest.TestCase):
    def test_build_v10_model(self):
        from vla_model.train_yolo_v10 import build_v10_model
        model = build_v10_model(seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
                                n_layers=1, ff_dim=64, dropout=0.0)
        self.assertEqual(model.version, "v10")


class MaskViewTests(unittest.TestCase):
    def test_last_mask_view_selects_last_raw_frame(self):
        from vla_model.visualize_yolo import select_mask_view
        masks = torch.arange(1 * 4 * 2 * 3 * 3, dtype=torch.float32).view(1, 4, 2, 3, 3)
        result = select_mask_view(masks, "last")
        expected = masks[:, :, -1]
        self.assertTrue(torch.equal(result, expected))

    def test_average_mask_view_preserves_v9_behavior(self):
        from vla_model.visualize_yolo import select_mask_view
        masks = torch.arange(1 * 4 * 2 * 3 * 3, dtype=torch.float32).view(1, 4, 2, 3, 3)
        result = select_mask_view(masks, "avg")
        expected = masks.mean(dim=2)
        self.assertTrue(torch.equal(result, expected))

    def test_unknown_mask_view_raises(self):
        from vla_model.visualize_yolo import select_mask_view
        masks = torch.randn(1, 4, 2, 3, 3)
        with self.assertRaises(ValueError):
            select_mask_view(masks, "invalid")


if __name__ == "__main__":
    unittest.main()
