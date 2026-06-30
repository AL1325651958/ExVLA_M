"""ExcavatorVLA: Vision-Language-Action model with Transformer for joint prediction.

v2 improvements:
  - Two-stream vision encoder (RGB + Elevation separate backbones)
  - Excavator ID embedding (per-model learnable bias)
  - Proprioception input (current qpos)
  - Action chunking (predict K future frames)
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
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    new_weight = self.backbone.conv1.weight
                    for c in range(min(3, in_channels)):
                        new_weight[:, c] = old_conv.weight[:, c % 3]
                    self.backbone.conv1.weight.copy_(new_weight)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self._out_features = in_features

        if in_features != hidden_dim:
            self.proj = nn.Linear(in_features, hidden_dim)
        else:
            self.proj = nn.Identity()

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class TwoStreamEncoder(nn.Module):
    """Two-tower vision encoder: separate ResNet-18 for RGB and Elevation, then concat."""

    def __init__(self, hidden_dim: int = 512, pretrained: bool = True):
        super().__init__()
        half_dim = hidden_dim // 2
        self.rgb_encoder = ImageEncoder(3, half_dim, pretrained)
        self.elev_encoder = ImageEncoder(3, half_dim, pretrained)

    def forward(self, rgb: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:       [B, 3, H, W]
            elevation: [B, 3, H, W]
        Returns:
            features:  [B, hidden_dim] — concat of both streams
        """
        rgb_feat = self.rgb_encoder(rgb)
        elev_feat = self.elev_encoder(elevation)
        return torch.cat([rgb_feat, elev_feat], dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding (used as fallback)."""

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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ExcavatorVLA(nn.Module):
    """VLA model v2: two-stream vision + proprioception + excavator ID → Transformer → action chunk.

    Input:
        rgb           [B, T, 3, H, W]
        elevation     [B, T, 3, H, W]
        qpos          [B, T, 4]     current joint state (proprioception)
        excavator_id  [B]           int: 0=75, 1=306, 2=490
    Output:
        action        [B, K, 4]     predicted joint positions for next K frames
    """

    def __init__(
        self,
        seq_len: int = 8,
        action_chunk: int = 5,       # predict next K frames
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
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim

        # Two-stream vision encoder (separate RGB + Elevation towers)
        self.vision_encoder = TwoStreamEncoder(hidden_dim, pretrained=pretrained)

        # Excavator ID embedding — per-excavator learnable bias
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)

        # Proprioception: project 4-DOF qpos → hidden_dim
        self.qpos_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.sinusoidal_pe = PositionalEncoding(hidden_dim, dropout=0.0)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=False, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head → outputs K frames × 4 DOF
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_chunk * 4),
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

    def forward(
        self,
        rgb: torch.Tensor,
        elevation: torch.Tensor,
        qpos: torch.Tensor = None,
        excavator_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns:
            action: [B, action_chunk, 4]
        """
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]

        # Flatten batch × time for vision encoder
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)
        features = self.vision_encoder(rgb_flat, elev_flat)  # [B*T, D]
        features = features.view(B, T, -1)                    # [B, T, D]

        # Add proprioception
        if qpos is not None:
            features = features + self.qpos_proj(qpos)

        # Add excavator ID embedding
        if excavator_id is not None:
            features = features + self.excv_embed(excavator_id).unsqueeze(1)

        # Position encoding
        if T <= self.seq_len:
            features = features + self.pos_embed[:, :T, :]
        else:
            pseudo = features.permute(1, 0, 2)
            pseudo = self.sinusoidal_pe(pseudo)
            features = pseudo.permute(1, 0, 2)

        # Transformer
        features = features.permute(1, 0, 2)  # [T, B, D]
        encoded = self.transformer(features)
        encoded = encoded.permute(1, 0, 2)    # [B, T, D]

        # Predict action chunk from last-frame feature
        last_feat = encoded[:, -1, :]          # [B, D]
        action = self.action_head(last_feat)   # [B, K * 4]
        action = action.view(B, self.action_chunk, 4)

        return action


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
