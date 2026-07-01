"""VLA model configuration."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data
    data_dir: str = "data/excavator-motion"
    seq_len: int = 8
    action_chunk: int = 1          # single step (delta prediction)
    img_size: int = 224            # can override with --img_size
    train_split: float = 0.9
    sample_ratio: float = 0.2

    # Model
    hidden_dim: int = 512
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 2048
    dropout: float = 0.1           # moderate regularization
    drop_path_rate: float = 0.05   # light stochastic depth
    pretrained: bool = True
    predict_delta: bool = True     # output Δqpos (change from last qpos)

    # Training
    batch_size: int = 32
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 3e-4     # light L2
    use_ema: bool = True
    ema_decay: float = 0.999
    smooth_loss_weight: float = 0.0  # no chunking, no smooth loss needed
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    device: str = "cuda"

    # Output
    output_dir: str = "output/checkpoints"
    log_interval: int = 10
    save_interval: int = 10
    num_workers: int = 0

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)
