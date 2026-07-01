"""ExcavatorVLA: two-stream vision + Transformer -> absolute joint angle prediction.

Pure vision at inference. qpos is privileged info (training only).
Output: next absolute qpos [B, 4] -- no delta, no anchor needed.
"""

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


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


class ExcavatorVLA(nn.Module):
    """VLA: pure vision -> absolute next joint position.

    qpos is privileged information: injected into Transformer during training
    only. At inference (model.eval()), prediction is purely vision-based.

    Input:
        rgb           [B, T, 3, H, W]
        elevation     [B, T, 3, H, W]
        qpos          [B, T, 4]     (training only, ignored at inference)
        excavator_id  [B]
    Output:
        next_qpos     [B, 4]        absolute joint angles (radians)
    """

    def __init__(
        self, seq_len=8, hidden_dim=512, n_heads=8, n_layers=4, ff_dim=2048,
        dropout=0.1, drop_path_rate=0.05, pretrained=True, num_excavators=4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.vision_encoder = TwoStreamEncoder(hidden_dim, pretrained=pretrained)
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)
        # Privileged: qpos injected into Transformer during training only
        self.qpos_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.sinusoidal_pe = PositionalEncoding(hidden_dim, dropout=0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=False, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.stoch_depth = DropPath(drop_path_rate)

        # Head: vision_feat + excavator_id -> absolute qpos
        head_in = hidden_dim + hidden_dim
        self.action_head = nn.Sequential(
            nn.Linear(head_in, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        for module in self.action_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)

    def forward(self, rgb, elevation, qpos=None, excavator_id=None):
        """
        Returns:
            next_qpos [B, 4] -- absolute joint angles (radians)
        """
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]

        # 1. Vision encoder (per-frame)
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)
        features = self.vision_encoder(rgb_flat, elev_flat).view(B, T, -1)

        # 2. qpos: privileged info, training only
        if qpos is not None and self.training:
            features = features + self.qpos_proj(qpos)

        # 3. Position encoding
        if T <= self.seq_len:
            features = features + self.pos_embed[:, :T, :]
        else:
            features = self.sinusoidal_pe(features.permute(1, 0, 2)).permute(1, 0, 2)

        # 4. Transformer
        features = features.permute(1, 0, 2)
        encoded = self.stoch_depth(self.transformer(features))
        encoded = encoded.permute(1, 0, 2)

        # 5. Head: vision + excavator ID -> absolute qpos
        vision_feat = encoded[:, -1, :]
        excv_feat = self.excv_embed(excavator_id)
        next_qpos = self.action_head(torch.cat([vision_feat, excv_feat], dim=-1))

        return next_qpos


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
