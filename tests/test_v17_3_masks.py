"""Focused tests for the V17.3 mask-regularization training variant."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from vla_model.model_yolo import load_compatible_state_dict
from vla_model.train_yolo_v17_1 import (
    accumulate_mask_diagnostics,
    build_v17_1_model,
    compute_mask_diagnostics,
    compute_mask_diversity_loss,
    format_mask_diagnostics,
    get_training_variant,
    save_checkpoint,
    validate,
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

    def test_rectangular_mask_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[B, 2, 4, T, G, G\]"):
            compute_mask_diversity_loss(
                torch.zeros(1, 2, 4, 1, 2, 3),
                mode="within_modality",
                margin=0.5,
            )

    def test_legacy_half_precision_matches_original_v17_1_math(self):
        torch.manual_seed(3)
        masks = torch.rand(2, 2, 4, 1, 2, 2, dtype=torch.float16)
        flat = masks.reshape(2, 8, -1)
        normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        eye = torch.eye(8, device=similarity.device).unsqueeze(0)
        expected = 0.5 * torch.relu(
            similarity * (1.0 - eye) - 0.3
        ).pow(2).mean()

        actual = compute_mask_diversity_loss(
            masks, mode="legacy_all_pairs", margin=0.3
        )

        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))


class V173DiagnosticTests(unittest.TestCase):
    def test_reports_area_centroid_and_cross_modal_similarity(self):
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

    def test_accumulator_is_sample_weighted(self):
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
        averaged = {key: value / count for key, value in totals.items()}

        self.assertEqual(count, 3)
        self.assertAlmostEqual(averaged["area"][0, 0].item(), 5.0 / 3.0)

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

    def test_validation_returns_sample_weighted_diagnostics(self):
        class DiagnosticModel(torch.nn.Module):
            out_dims = (2, 2, 2, 2)

            def forward(self, rgb, elevation, qpos, excavator_id):
                batch = rgb.shape[0]
                raw = torch.tensor(
                    [0.0, 1.0] * 4, dtype=rgb.dtype, device=rgb.device
                ).view(1, 8).expand(batch, -1)
                masks = torch.zeros(batch, 2, 4, 1, 2, 2, device=rgb.device)
                masks[:, :, 0, :, 0, 0] = 1.0
                return raw, masks.mean(dim=3), masks

            @staticmethod
            def decode_action(raw):
                pairs = raw.view(-1, 4, 2)
                return torch.atan2(pairs[..., 0], pairs[..., 1])

        batch = {
            "rgb": torch.zeros(1, 1, 3, 2, 2),
            "elevation": torch.zeros(1, 1, 3, 2, 2),
            "qpos": torch.zeros(1, 1, 4),
            "excavator_id": torch.zeros(1, dtype=torch.long),
            "action": torch.zeros(1, 1, 4),
        }
        config = SimpleNamespace(
            device="cpu",
            v17_swing_loss_weight=2.0,
            v17_mask_diagnostics=True,
        )

        metrics = validate(
            DiagnosticModel(), [batch], torch.nn.MSELoss(), config
        )

        self.assertIn("mask_diagnostics", metrics)
        self.assertAlmostEqual(
            metrics["mask_diagnostics"]["area"][0][0], 0.25
        )


class V173ArtifactTests(unittest.TestCase):
    def test_checkpoint_uses_v17_3_filename_and_metadata(self):
        model = torch.nn.Linear(2, 1)
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
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )

        self.assertEqual(Path(path).name, "yolo_v17_3_checkpoint_best.pt")
        self.assertEqual(checkpoint["model_version"], "v17.3")

    def test_entry_point_selects_v17_3(self):
        import vla_model.train_yolo_v17_3 as entry_point

        with mock.patch.object(
            entry_point, "train_v17_1_main", return_value=17
        ) as run:
            self.assertEqual(entry_point.main(), 17)
        run.assert_called_once_with(training_version="v17.3")

    def test_visualizer_maps_v17_3_to_v17_1_topology(self):
        from vla_model.visualize_yolo import detect_version

        result = detect_version(
            {"joint_logit_bias", "temporal_mask_mixer.layers.0.weight"},
            checkpoint_version="v17.3",
        )

        self.assertEqual(result, ("V17.3", True, "v17.1"))

    def test_v17_3_state_dict_loads_without_skipped_tensors(self):
        from vla_model.visualize_yolo import detect_version

        source = build_v17_1_model(
            seq_len=2,
            img_size=32,
            hidden_dim=32,
            n_heads=4,
            n_layers=1,
            ff_dim=64,
            dropout=0.0,
            pretrained=False,
            num_excavators=3,
        )
        checkpoint = {
            "model_version": "v17.3",
            "model_state_dict": source.state_dict(),
        }
        _, _, model_version = detect_version(
            set(checkpoint["model_state_dict"]),
            checkpoint_version=checkpoint["model_version"],
        )
        target = build_v17_1_model(
            seq_len=2,
            img_size=32,
            hidden_dim=32,
            n_heads=4,
            n_layers=1,
            ff_dim=64,
            dropout=0.0,
            pretrained=False,
            num_excavators=3,
        )

        loaded, skipped = load_compatible_state_dict(
            target, checkpoint["model_state_dict"]
        )

        self.assertEqual(model_version, "v17.1")
        self.assertEqual(skipped, 0)
        self.assertEqual(loaded, len(checkpoint["model_state_dict"]))


if __name__ == "__main__":
    unittest.main()
