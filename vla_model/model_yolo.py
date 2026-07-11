"""ExcavatorVLA-YOLO: spatio-temporal Transformer on grid features.

Architecture (Scheme 3):
  1. Per-frame: ResNet-18 extracts 7×7 spatial feature grid  [T, 49, D]
  2. Two-stream: independent ResNet for RGB and Elevation, concat at grid level
  3. All T×49 tokens + spatial(2D) + temporal(1D) pos encoding
  4. Transformer Encoder → joint spatio-temporal attention
  5. CLS token readout → MLP head → Δqpos [B, 4]
"""

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


# ---------------------------------------------------------------------------
#  Spatial grid backbone  (ResNet-18 strip to layer4, 7×7 grid)
# ---------------------------------------------------------------------------

class GridBackbone(nn.Module):
    """ResNet-18 → 7×7 spatial grid of features (before avgpool)."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    nw = backbone.conv1.weight
                    for c in range(min(3, in_channels)):
                        nw[:, c] = old_conv.weight[:, c % 3]
                    backbone.conv1.weight.copy_(nw)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # /4  → 56×56
        self.layer2 = backbone.layer2   # /8  → 28×28
        self.layer3 = backbone.layer3   # /16 → 14×14
        self.layer4 = backbone.layer4   # /32 → 7×7

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W] → grid: [B, 512, 7, 7]"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class TwoStreamGrid(nn.Module):
    """Two-tower: separate GridBackbone for RGB and Elevation, fuse at grid level."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.rgb_backbone  = GridBackbone(3, pretrained)
        self.elev_backbone = GridBackbone(3, pretrained)
        # After concat: 512+512 = 1024 channels per grid cell

    def forward(self, rgb: torch.Tensor, elevation: torch.Tensor):
        """Returns [B, 7, 7, 1024] grid."""
        grid_rgb  = self.rgb_backbone(rgb)        # [B, 512, 7, 7]
        grid_elev = self.elev_backbone(elevation)  # [B, 512, 7, 7]
        grid = torch.cat([grid_rgb, grid_elev], dim=1)  # [B, 1024, 7, 7]
        return grid.permute(0, 2, 3, 1)  # → [B, 7, 7, C]


# ---------------------------------------------------------------------------
#  Position encoding  (2D spatial + 1D temporal)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, device=None):
    """Fixed 2D sin/cos embed for grid_size×grid_size."""
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin/cos")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return torch.from_numpy(pos_embed).float().to(device)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """grid: [2, 1, H, W]. Returns [1, H*W, embed_dim]."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """pos: [1, H, W] flattened. Returns [H*W, embed_dim]."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


import numpy as np


class SpatioTemporalPosEmbed(nn.Module):
    """Learnable temporal pos + fixed 2D spatial sin/cos pos for grid tokens."""

    def __init__(self, max_frames: int, grid_size: int, grid_dim: int):
        super().__init__()
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.temp_embed = nn.Parameter(torch.randn(1, max_frames, 1, 1, 1) * 0.02)

        # Fixed 2D spatial embed  [1, H*W, D]
        grid_emb = get_2d_sincos_pos_embed(grid_dim, grid_size)
        if grid_emb.dim() == 2:
            grid_emb = grid_emb.unsqueeze(0)
        self.register_buffer("spatial_embed", grid_emb)

    def forward(self, features: torch.Tensor):
        """features: [B, T, H, W, D]. Returns [B, T, H, W, D] with pos embed."""
        B, T, H, W, D = features.shape
        pos_spatial = self.spatial_embed[:, : H * W, :].view(1, 1, H, W, D)  # [1, 1, H, W, D]
        pos_temp = self.temp_embed[:, :T, :, :, :]  # [1, T, 1, 1, 1]
        return features + pos_spatial + pos_temp


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class ExcavatorVLAYolo(nn.Module):
    """YOLO-style spatio-temporal model for excavator joint prediction.

    Input:
        rgb          [B, T, 3, H, W]
        elevation    [B, T, 3, H, W]
        qpos         [B, T, 4]     proprioception
        excavator_id [B]           0=75, 1=306, 2=490
    Output:
        delta         [B, 4]       predicted Δqpos
    """

    def __init__(
        self,
        seq_len: int = 8,
        img_size: int = 224,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
        num_excavators: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        # ── Spatial backbone ──
        grid_size = img_size // 32  # 224 → 7, 384 → 12
        grid_dim_raw = 1024          # RGB(512) + Elev(512)
        self.grid_size = grid_size
        self.grid_dim_raw = grid_dim_raw
        self.spatial_net = TwoStreamGrid(pretrained=pretrained)

        # ── Grid → token projection ──
        self.grid_proj = nn.Linear(grid_dim_raw, hidden_dim)

        # ── Proprioception projector ──
        self.qpos_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # ── Excavator ID embedding ──
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)

        # ── Position encoding ──
        self.pos_embed = SpatioTemporalPosEmbed(seq_len, grid_size, hidden_dim)

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ── Transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Readout head ──
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        for module in self.delta_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)

    def forward(self, rgb, elevation, qpos=None, excavator_id=None):
        """Returns delta [B, 4]."""
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]
        G = self.grid_size  # grid size (e.g. 7)
        D = self.hidden_dim

        # ── Step 1: Per-frame spatial grid ──
        # Fold batch+time for backbone
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)

        grid = self.spatial_net(rgb_flat, elev_flat)            # [B*T, G, G, 1024]
        grid = self.grid_proj(grid)                              # [B*T, G, G, D]
        grid = grid.view(B, T, G, G, D)                          # [B, T, G, G, D]

        # ── Step 2: Add proprioception (per-frame, per-grid-cell) ──
        if qpos is not None:
            qpos_feat = self.qpos_proj(qpos)                     # [B, T, D]
            grid = grid + qpos_feat.unsqueeze(2).unsqueeze(2)    # broadcast over grid

        # ── Step 3: Position encoding ──
        grid = self.pos_embed(grid)                              # [B, T, G, G, D]

        # ── Step 4: Flatten to tokens ──
        tokens = grid.reshape(B, T * G * G, D)                   # [B, T×G×G, D]

        # ── Step 5: Add excavator embedding ──
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # ── Step 6: Prepend CLS token ──
        cls_tokens = self.cls_token.expand(B, -1, -1)            # [B, 1, D]
        tokens = torch.cat([cls_tokens, tokens], dim=1)          # [B, 1+T×G×G, D]

        # ── Step 7: Transformer ──
        encoded = self.transformer(tokens)                       # [B, N, D]

        # ── Step 8: CLS readout → delta ──
        cls_feat = encoded[:, 0, :]                              # [B, D]
        delta = self.delta_head(cls_feat)                        # [B, 4]

        return delta


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
