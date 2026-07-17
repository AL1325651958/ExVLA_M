"""Regression tests for V17.1 Swing supervision and model routing."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from vla_model.model_yolo import ExcavatorVLAYolo
from vla_model.train_yolo_v17_1 import (
    _compute_r2_circular,
    _upgrade_legacy_v17_1_state_dict,
    _weighted_sincos_loss,
    save_checkpoint,
    restore_training_state,
    train_epoch,
    update_ema,
)


def build_small_v17_1():
    return ExcavatorVLAYolo(
        seq_len=2,
        img_size=32,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        ff_dim=64,
        dropout=0.0,
        pretrained=False,
        num_excavators=3,
        version="v17.1",
    )


class CountingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value):
        self.calls.append(tuple(value.shape))
        return value


class V171ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = build_small_v17_1().eval()
        self.rgb = torch.randn(1, 2, 3, 32, 32)
        self.elevation = torch.randn(1, 2, 3, 32, 32)
        self.excavator_id = torch.zeros(1, dtype=torch.long)

    def test_uses_four_independent_v17_mask_heads(self):
        self.assertIsInstance(self.model.mask_linear1, nn.ModuleList)
        self.assertIsInstance(self.model.mask_linear2, nn.ModuleList)
        self.assertEqual(len(self.model.mask_linear1), 4)
        self.assertIsNone(self.model.swing_mask_linear1)
        self.assertEqual(tuple(self.model.joint_logit_bias.shape), (4,))

    def test_training_aux_outputs_are_periodic_pose_and_independent_velocity(self):
        with torch.no_grad():
            outputs = self.model(
                self.rgb, self.elevation, None, self.excavator_id, return_aux=True
            )
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[3].shape, (1, 8))
        self.assertEqual(outputs[4].shape, (1, 2))

    def test_temporal_mixer_runs_on_both_modalities_before_masks(self):
        counter = CountingIdentity()
        self.model.temporal_mask_mixer = counter
        with torch.no_grad():
            self.model(self.rgb, self.elevation, None, self.excavator_id)
        self.assertEqual(len(counter.calls), 2)
        self.assertEqual(counter.calls[0], (1, 2, 2, 2, 32))
        self.assertEqual(counter.calls[1], (1, 2, 2, 2, 32))

    def test_inference_remains_invariant_to_qpos(self):
        with torch.no_grad():
            left = self.model(
                self.rgb, self.elevation, torch.zeros(1, 2, 4), self.excavator_id
            )
            right = self.model(
                self.rgb, self.elevation, torch.full((1, 2, 4), 91.0), self.excavator_id
            )
        for left_tensor, right_tensor in zip(left, right):
            self.assertTrue(torch.equal(left_tensor, right_tensor))


class V171TrainingUtilityTests(unittest.TestCase):
    def test_incompatible_optimizer_does_not_restore_stale_scheduler(self):
        source_model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-4)
        source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1)
        source_optimizer.step()
        source_scheduler.step()
        checkpoint = {
            "optimizer_state_dict": source_optimizer.state_dict(),
            "scheduler_state_dict": source_scheduler.state_dict(),
        }

        target_model = nn.Linear(2, 1)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=3e-4)
        target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=5)
        initial_scheduler_epoch = target_scheduler.last_epoch

        restored = restore_training_state(
            target_optimizer, None, target_scheduler, checkpoint
        )
        self.assertFalse(restored)
        self.assertEqual(target_scheduler.last_epoch, initial_scheduler_epoch)

    def test_complete_training_batch_runs_with_ema(self):
        model = build_small_v17_1()
        ema_model = deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cpu")
        config = SimpleNamespace(
            device="cpu",
            v17_pose_aux_weight=0.05,
            v17_vel_aux_weight=0.10,
            v17_swing_loss_weight=2.0,
            grad_clip=1.0,
            ema_decay=0.9,
            log_interval=100,
            epochs=1,
        )
        batch = {
            "rgb": torch.randn(1, 2, 3, 32, 32),
            "elevation": torch.randn(1, 2, 3, 32, 32),
            "qpos": torch.randn(1, 2, 4) * 0.1,
            "excavator_id": torch.zeros(1, dtype=torch.long),
            "action": torch.randn(1, 1, 4) * 0.1,
        }
        metrics = train_epoch(
            model, [batch], optimizer, scaler, scheduler, nn.MSELoss(),
            config, epoch=0, ema_model=ema_model,
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(len(metrics["mae"]), 4)

    def test_visualizer_honors_v17_1_checkpoint_metadata(self):
        from vla_model.visualize_yolo import detect_version

        version_tag, is_dual, model_version = detect_version(
            {"rgb_proj.weight", "temporal_mask_mixer.layers.0.weight"},
            checkpoint_version="v17.1",
        )
        self.assertEqual((version_tag, is_dual, model_version), ("V17.1", True, "v17.1"))

    def test_visualizer_infers_exact_encoder_depth(self):
        from vla_model.visualize_yolo import infer_transformer_config

        state = {
            f"encoder.layers.{index}.self_attn.in_proj_weight": torch.zeros(96, 32)
            for index in range(3)
        }
        state["encoder.layers.0.linear1.weight"] = torch.zeros(64, 32)
        hidden_dim, n_layers, ff_dim = infer_transformer_config(state)
        self.assertEqual((hidden_dim, n_layers, ff_dim), (32, 3, 64))

    def test_swing_r2_uses_wrapped_squared_deviation_from_circular_mean(self):
        labels = [torch.pi - 0.1, -torch.pi + 0.1]
        ss_res = torch.tensor([0.0, 0.0, 0.0, 0.01]).numpy()
        sum_y = torch.zeros(4).numpy()
        sum_y2 = torch.ones(4).numpy()
        r2, _ = _compute_r2_circular(ss_res, sum_y, sum_y2, 2, labels)
        self.assertAlmostEqual(r2[3], 0.5, places=4)

    def test_legacy_shared_masks_are_migrated_to_four_independent_heads(self):
        legacy = {
            "mask_linear1.weight": torch.full((3, 4), 1.0),
            "mask_linear1.bias": torch.full((3,), 2.0),
            "mask_linear2.weight": torch.full((1, 3), 3.0),
            "mask_linear2.bias": torch.full((1,), 4.0),
            "swing_mask_linear1.weight": torch.full((3, 4), 5.0),
            "swing_mask_linear1.bias": torch.full((3,), 6.0),
            "swing_mask_linear2.weight": torch.full((1, 3), 7.0),
            "swing_mask_linear2.bias": torch.full((1,), 8.0),
        }
        upgraded = _upgrade_legacy_v17_1_state_dict(legacy)
        for joint in range(3):
            self.assertTrue(torch.equal(
                upgraded[f"mask_linear1.{joint}.weight"], legacy["mask_linear1.weight"]
            ))
        self.assertTrue(torch.equal(
            upgraded["mask_linear1.3.weight"], legacy["swing_mask_linear1.weight"]
        ))
        self.assertTrue(torch.equal(
            upgraded["mask_linear2.3.bias"], legacy["swing_mask_linear2.bias"]
        ))

    def test_weighted_loss_emphasizes_swing(self):
        prediction = torch.zeros(1, 8)
        target = torch.zeros(1, 8)
        prediction[:, 6:] = 1.0
        unweighted = _weighted_sincos_loss(prediction, target, swing_weight=1.0)
        weighted = _weighted_sincos_loss(prediction, target, swing_weight=2.0)
        self.assertGreater(weighted.item(), unweighted.item())

    def test_ema_updates_each_parameter_toward_live_model(self):
        live = nn.Linear(2, 1, bias=False)
        ema = deepcopy(live)
        with torch.no_grad():
            live.weight.fill_(2.0)
            ema.weight.zero_()
        update_ema(ema, live, decay=0.75)
        self.assertTrue(torch.allclose(ema.weight, torch.full_like(ema.weight, 0.5)))

    def test_checkpoint_persists_best_metrics(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(output_dir=tmp)
            path = save_checkpoint(
                model,
                optimizer,
                scaler,
                scheduler,
                epoch=4,
                metrics={"loss": 0.3, "r2": [0.0, 0.0, 0.0, 0.8]},
                config=config,
                best_loss=0.2,
                best_swing_r2=0.85,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["best_loss"], 0.2)
        self.assertEqual(checkpoint["best_swing_r2"], 0.85)


if __name__ == "__main__":
    unittest.main()
