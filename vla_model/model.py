"""ExcavatorVLA: vision-first architecture with privileged qpos modulation.

Encoder options: Transformer (default) or Mamba (O(n) state-space model).
Vision (RGB+Elevation) is the primary predictor. qpos acts as a small
residual correction at the head level during training only.
Inference: pure vision, zero qpos dependency.
"""

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


# ---------------------------------------------------------------------------
#  Components
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Encode a single image stream using ResNet-18."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_weight = self.backbone.conv1.weight
                    for c in range(min(3, in_channels)):
                        new_weight[:, c] = old_conv.weight[:, c % 3]
                    self.backbone.conv1.weight.copy_(new_weight)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class TwoStreamEncoder(nn.Module):
    """Two-tower: separate ResNet-18 for RGB and Elevation, then concat."""

    def __init__(self, hidden_dim: int = 512, pretrained: bool = True):
        super().__init__()
        half_dim = hidden_dim // 2
        self.rgb_encoder = ImageEncoder(3, half_dim, pretrained)
        self.elev_encoder = ImageEncoder(3, half_dim, pretrained)

    def forward(self, rgb: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rgb_encoder(rgb), self.elev_encoder(elevation)], dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding (used by Transformer encoder)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0)])


class DropPath(nn.Module):
    """Stochastic Depth for Transformer regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ---------------------------------------------------------------------------
#  Mamba-style block  (selective state-space, O(n) complexity)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Selective state-space block (Mamba-style) with diagonal SSM recurrence."""

    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        inner = int(d_model * expand)

        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=inner, out_channels=inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=inner,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        )
        self.B_proj = nn.Linear(inner, d_state, bias=False)
        self.C_proj = nn.Linear(inner, d_state, bias=False)
        self.dt_proj = nn.Linear(inner, inner, bias=True)
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D_in = x.shape
        inner = self.in_proj.weight.shape[0] // 2
        d_state = self.d_state
        residual = x

        proj = self.in_proj(x)                      # [B, T, inner*2]
        z, x_sig = proj.chunk(2, dim=-1)            # [B, T, inner] each

        x_conv = self.conv1d(x_sig.transpose(1, 2))[:, :, :T]
        x_conv = F.silu(x_conv.transpose(1, 2))      # [B, T, inner]

        dt = F.softplus(self.dt_proj(x_conv))        # [B, T, inner]
        B_sel = self.B_proj(x_conv)                  # [B, T, d_state]
        C_sel = self.C_proj(x_conv)                  # [B, T, d_state]

        A = -torch.exp(self.A_log)                   # [d_state]

        # ZOH discretization + selective SSM scan
        dt_A = dt.unsqueeze(-1) * A                  # [B, T, inner, d_state]
        A_bar = torch.exp(dt_A)                       # element-wise exp
        B_bar = dt.unsqueeze(-1) * B_sel.unsqueeze(2) # [B, T, inner, d_state]

        # Selective scan (sequential — could be parallelized with associative scan)
        h = torch.zeros(B, inner, d_state, device=x.device, dtype=x.dtype)
        y_ssm = torch.zeros(B, T, inner, device=x.device, dtype=x.dtype)
        for t_step in range(T):
            h = A_bar[:, t_step] * h + B_bar[:, t_step]
            y_ssm[:, t_step] = (h * C_sel[:, t_step].unsqueeze(2)).sum(dim=-1)

        y = y_ssm + self.D * x_conv                   # [B, T, inner]
        y = y * F.silu(z)                              # gating
        y = self.dropout(self.out_proj(y))
        return self.norm(residual + y)


class MambaEncoder(nn.Module):
    """Stack of MambaBlocks — drop-in replacement for TransformerEncoder."""

    def __init__(self, d_model: int = 512, n_layers: int = 4, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → y: [B, T, D]"""
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
#  Joint encoding helpers
# ---------------------------------------------------------------------------

def encode_joints_sincos(qpos_rad: torch.Tensor) -> torch.Tensor:
    sin = torch.sin(qpos_rad)
    cos = torch.cos(qpos_rad)
    return torch.cat([sin, cos], dim=-1)


def decode_joints_sincos(sincos: torch.Tensor) -> torch.Tensor:
    sin, cos = sincos.chunk(2, dim=-1)
    return torch.atan2(sin, cos)


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class ExcavatorVLA(nn.Module):
    """Vision-first VLA with swappable temporal encoder.

    Architecture:
        RGB + Elev → ResNet → [Transformer | Mamba] → vision_feat ─┐
                                                                    ├→ base_pred
        excv_id → per-head selection ──────────────────────────────┘
        qpos → modulation MLP → correction (training only) ──────┘
                                                                    ↓
                                                          final_pred [B, K, out]
    """

    def __init__(
        self, seq_len=8, hidden_dim=512, n_heads=8, n_layers=4, ff_dim=2048,
        dropout=0.1, drop_path_rate=0.05, pretrained=True, num_excavators=4,
        qpos_mode: str = "modulation", qpos_drop_prob=0.3,
        encoder_type: str = "transformer",
        mamba_d_state: int = 0, mamba_d_conv: int = 4, mamba_expand: int = 2,
        use_sincos: bool = False,
        use_sincos_output: bool = False,
        action_chunk: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.qpos_mode = qpos_mode
        self.qpos_drop_prob = qpos_drop_prob
        self.encoder_type = encoder_type
        self.use_sincos = use_sincos
        self.use_sincos_output = use_sincos_output
        self.action_chunk = action_chunk

        qpos_dim = 8 if use_sincos else 4
        out_dim = 8 if use_sincos_output else 4

        # ── Vision backbone ──
        self.vision_encoder = TwoStreamEncoder(hidden_dim, pretrained=pretrained)
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)

        # ── Temporal encoder ──
        if encoder_type == "transformer":
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
            self.sinusoidal_pe = PositionalEncoding(hidden_dim, dropout=0.0)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, activation='gelu', batch_first=False, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.stoch_depth = DropPath(drop_path_rate)
        elif encoder_type == "mamba":
            self.encoder = MambaEncoder(
                d_model=hidden_dim, n_layers=n_layers,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ── Per-excavator prediction heads ──
        head_in = hidden_dim
        head_out = action_chunk * out_dim
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, head_out),
            )
            for _ in range(num_excavators)
        ])

        # ── qpos components ──
        if qpos_mode == "none":
            self.qpos_mod = None
        elif qpos_mode == "modulation":
            self.qpos_mod = nn.Sequential(
                nn.Linear(qpos_dim, 32), nn.GELU(), nn.Linear(32, out_dim),
            )
        elif qpos_mode == "transformer":
            self.qpos_proj = nn.Sequential(
                nn.Linear(qpos_dim, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown qpos_mode: {qpos_mode}")

        self._init_weights()

    def _init_weights(self):
        if self.encoder_type == "transformer":
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.5)
        for head in self.action_heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)
        if self.qpos_mode == "modulation":
            for module in self.qpos_mod:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif self.qpos_mode == "transformer":
            for module in self.qpos_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, rgb, elevation, qpos=None, excavator_id=None):
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]

        # 1. Vision encoder
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)
        features = self.vision_encoder(rgb_flat, elev_flat).view(B, T, -1)

        # 2. qpos injection (transformer mode) — training only
        if self.qpos_mode == "transformer" and qpos is not None and self.training:
            qpos_enc = encode_joints_sincos(qpos) if self.use_sincos else qpos
            qpos_feat = self.qpos_proj(qpos_enc)
            if self.qpos_drop_prob > 0:
                mask = (torch.rand(B, 1, 1, device=qpos.device) > self.qpos_drop_prob).float()
                qpos_feat = qpos_feat * mask
            features = features + qpos_feat

        # 3. Temporal encoder
        if self.encoder_type == "transformer":
            if T <= self.seq_len:
                features = features + self.pos_embed[:, :T, :]
            else:
                features = self.sinusoidal_pe(features.permute(1, 0, 2)).permute(1, 0, 2)
            features = features.permute(1, 0, 2)
            encoded = self.stoch_depth(self.encoder(features))
            encoded = encoded.permute(1, 0, 2)
        else:
            encoded = self.encoder(features)

        # 4. Per-excavator head
        vision_feat = encoded[:, -1, :]
        K = self.action_chunk
        out_dim = 8 if self.use_sincos_output else 4
        base_pred = torch.zeros(B, K * out_dim, device=vision_feat.device, dtype=vision_feat.dtype)
        for excv_idx in range(len(self.action_heads)):
            mask = (excavator_id == excv_idx)
            if mask.any():
                head_out = self.action_heads[excv_idx](vision_feat[mask].float())
                base_pred[mask] = head_out.to(vision_feat.dtype)

        if K == 1:
            base_pred = base_pred.view(B, out_dim)
        else:
            base_pred = base_pred.view(B, K, out_dim)

        # 5. qpos modulation (modulation mode) — training only
        if self.qpos_mode == "modulation" and qpos is not None and self.training:
            qpos_last = encode_joints_sincos(qpos[:, -1, :]) if self.use_sincos else qpos[:, -1, :]
            correction = self.qpos_mod(qpos_last)
            if self.qpos_drop_prob > 0:
                mask = (torch.rand(B, 1, device=qpos.device) > self.qpos_drop_prob).float()
                correction = correction * mask
            if K == 1:
                return base_pred + correction
            else:
                return base_pred + correction.unsqueeze(1)
        else:
            return base_pred


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
