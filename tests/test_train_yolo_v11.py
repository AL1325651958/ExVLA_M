"""Focused contracts for the V11 training entrypoint."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class V11TrainingEntrypointTests(unittest.TestCase):
    def test_build_v11_model_selects_v11_architecture(self):
        from vla_model.train_yolo_v11 import build_v11_model
        model = build_v11_model(seq_len=2, img_size=32, hidden_dim=32, n_heads=4,
                                n_layers=1, ff_dim=64, dropout=0.0)
        self.assertEqual(model.version, "v11")

    def test_each_optimizer_batch_advances_scheduler_once(self):
        from vla_model.train_yolo_v11 import step_batch_scheduler

        class CountingScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        scheduler = CountingScheduler()
        for _ in range(3):
            step_batch_scheduler(scheduler)
        self.assertEqual(scheduler.steps, 3)

    def test_checkpoint_metadata_identifies_v11(self):
        from vla_model.train_yolo_v11 import save_checkpoint

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(output_dir=tmpdir)
            save_checkpoint(model, optimizer, scaler, 0, {"loss": 1.0}, config, is_best=True)
            checkpoint = torch.load(Path(tmpdir) / "yolo_checkpoint_best.pt", weights_only=False)
        self.assertEqual(checkpoint["model_version"], "v11")

    def test_validation_never_passes_qpos_into_model_forward(self):
        from vla_model.train_yolo_v11 import validate

        class RecordingModel(torch.nn.Module):
            out_dims = (2, 2, 2, 2)

            def __init__(self):
                super().__init__()
                self.qpos_arguments = []

            def forward(self, rgb, elevation, qpos=None, excavator_id=None):
                self.qpos_arguments.append(qpos)
                return (torch.zeros(rgb.size(0), 8), torch.zeros(rgb.size(0), 4, 1, 1),
                        torch.zeros(rgb.size(0), 4, 1, 1, 1))

            def decode_action(self, raw):
                return torch.zeros(raw.size(0), 4)

        model = RecordingModel()
        batch = {"rgb": torch.zeros(1, 1, 3, 4, 4), "elevation": torch.zeros(1, 1, 3, 4, 4),
                 "qpos": torch.full((1, 1, 4), 7.0), "excavator_id": torch.zeros(1, dtype=torch.long),
                 "action": torch.zeros(1, 1, 4)}
        validate(model, [batch], torch.nn.MSELoss(), type("C", (), {"device": "cpu"})())
        self.assertEqual(model.qpos_arguments, [None])


if __name__ == "__main__":
    unittest.main()
