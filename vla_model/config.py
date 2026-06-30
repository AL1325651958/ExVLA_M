"""VLA model configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data
    data_dir: str = "data/excavator-motion"
    seq_len: int = 8              # frames per sequence
    action_chunk: int = 1         # predict next N frames (1 = single step)
    img_size: int = 160           # resize to (img_size, img_size)
    train_split: float = 0.9      # train/val split ratio

    # Model
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 1024
    dropout: float = 0.1
    pretrained: bool = True       # use pretrained ResNet-18

    # Training
    batch_size: int = 8
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    device: str = "cuda"

    # Output
    output_dir: str = "output/checkpoints"
    log_interval: int = 10        # steps per log
    save_interval: int = 10       # epochs per checkpoint
    num_workers: int = 0  # 0 = main process only (data preloaded in memory)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)
