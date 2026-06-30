"""ExcavatorVLA: Vision-Language-Action model with Transformer for joint prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


class ImageEncoder(nn.Module):
    """Encode RGB + Elevation images using a modified ResNet-18 (early fusion)."""

    def __init__(self, hidden_dim: int = 512, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace first conv to accept 6-channel input (RGB 3 + Elevation 3)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            6, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new channels: average the pretrained RGB weights
        if pretrained:
            with torch.no_grad():
                rgb_weight = old_conv.weight  # [64, 3, 7, 7]
                new_weight = self.backbone.conv1.weight  # [64, 6, 7, 7]
                new_weight[:, :3] = rgb_weight
                new_weight[:, 3:] = rgb_weight  # Copy RGB weights for elevation
                self.backbone.conv1.weight.copy_(new_weight)

        # Replace FC layer with identity (use backbone as feature extractor)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self._out_features = in_features  # 512 for ResNet-18

        # Optional projection to hidden_dim if different
        if in_features != hidden_dim:
            self.proj = nn.Linear(in_features, hidden_dim)
        else:
            self.proj = nn.Identity()

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, rgb: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:       [B, 3, H, W]
            elevation: [B, 3, H, W]
        Returns:
            features:  [B, out_features]
        """
        x = torch.cat([rgb, elevation], dim=1)  # [B, 6, H, W]
        x = self.backbone(x)                     # [B, 512]
        x = self.proj(x)
        return x


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
        self.register_buffer('pe', pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, D]
        Returns:
            x + sin/cos positional encoding
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ExcavatorVLA(nn.Module):
    """VLA model: visual encoder → Transformer → joint position prediction.

    Input:  sequence of (RGB + Elevation) images    [B, T, 3, H, W] each
    Output: predicted joint positions                [B, action_chunk, 4]
    """

    def __init__(
        self,
        seq_len: int = 8,
        action_chunk: int = 1,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim

        # Vision encoder
        self.vision_encoder = ImageEncoder(hidden_dim, pretrained=pretrained)

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Fallback sinusoidal encoding for variable-length inference
        self.sinusoidal_pe = PositionalEncoding(hidden_dim, dropout=0.0)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False,  # [T, B, D]
            norm_first=True,    # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_chunk * 4),
        )

        # Initialize weights
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

    def encode_images(self, rgb: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """Encode a batch of image pairs.

        Args:
            rgb:       [B, 3, H, W]
            elevation: [B, 3, H, W]
        Returns:
            features:  [B, hidden_dim]
        """
        return self.vision_encoder(rgb, elevation)

    def forward(
        self,
        rgb: torch.Tensor,
        elevation: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            rgb:       [B, T, 3, H, W]
            elevation: [B, T, 3, H, W]
            use_cache: if True, process frames incrementally (inference)
        Returns:
            action:    [B, action_chunk, 4] predicted joint positions
        """
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]

        # Encode all frames
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)
        features = self.vision_encoder(rgb_flat, elev_flat)  # [B*T, D]
        features = features.view(B, T, -1)                    # [B, T, D]

        # Add position encoding
        if T <= self.seq_len:
            features = features + self.pos_embed[:, :T, :]
        else:
            # Use sinusoidal for longer sequences at inference
            pseudo = features.permute(1, 0, 2)  # [T, B, D]
            pseudo = self.sinusoidal_pe(pseudo)
            features = pseudo.permute(1, 0, 2)  # [B, T, D]

        # Transformer expects [T, B, D]
        features = features.permute(1, 0, 2)  # [T, B, D]
        encoded = self.transformer(features)   # [T, B, D]
        encoded = encoded.permute(1, 0, 2)    # [B, T, D]

        # Predict action from last-frame feature
        last_feat = encoded[:, -1, :]          # [B, D]
        action = self.action_head(last_feat)   # [B, action_chunk * 4]
        action = action.view(B, self.action_chunk, 4)

        return action

    def predict_single_step(self, rgb: torch.Tensor) -> torch.Tensor:
        """Quick prediction for a single image (no temporal context)."""
        return self.action_head(rgb)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
