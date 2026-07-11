"""VLA model configuration."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data
    data_dir: str = "data/excavator-motion"
    seq_len: int = 8
    action_chunk: int = 1          # single step (YOLO model default)
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
    predict_delta: bool = False       # absolute qpos prediction
    qpos_mode: str = "modulation"      # "none" | "modulation" | "transformer"
    qpos_drop_prob: float = 0.3        # probability of masking qpos during training
    qpos_drop_schedule: bool = False   # linear ramp: drop_prob increases over epochs
    qpos_drop_start: float = 0.0       #   epoch 0 drop probability
    qpos_drop_end: float = 1.0         #   final-epoch drop probability (1.0 = pure vision)
    use_sincos: bool = False               # encode input qpos as [sin(θ), cos(θ)] pairs
    use_sincos_output: bool = False        # predict [sin(θ), cos(θ)] — loss in circular space
    mamba_d_state: int = 0                # 0=Conv1d only, 4=minimal SSM, 16=full SSM

    # Training
    batch_size: int = 32
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 3e-4     # light L2
    use_ema: bool = True
    ema_decay: float = 0.999
    smooth_loss_weight: float = 0.05   # temporal smoothness: penalize |Δt+1 - Δt|
    consistency_loss_weight: float = 0.05  # overlapping predictions agree
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
