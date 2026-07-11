"""ExcavatorVLA-YOLO: authentic YOLO architecture adapted for excavator control.

Real YOLO elements:
  1. CSPDarknet backbone → 3-scale feature maps (P3/P4/P5)
  2. FPN+PAN neck — top-down + bottom-up multi-scale fusion
  3. SpatialConv prediction head — conv instead of linear, grid-aware
  4. Per-frame spatial features → Transformer for temporal fusion

Architecture:
  RGB+Elev → CSPDarknet → [P3:56×56, P4:28×28, P5:14×14]
           → FPN+PAN  → [N3:56×56, N4:28×28, N5:14×14]
           → SpatialConv Head → 7×7 grid × D per frame
           → 8×49 tokens + CLS → Transformer → sin/cos Δ joints [B,8]
"""

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math
import numpy as np


# ============================================================================
# 1. CSPDarknet-style backbone (light, fast, strong gradients)
# ============================================================================

class ConvBNSiLU(nn.Module):
    """Conv → BN → SiLU  (YOLO's basic building block)."""
    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPLayer(nn.Module):
    """Cross-Stage Partial layer — split channels, process half, concat+fuse."""

    def __init__(self, in_c, out_c, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden = int(out_c * expansion)
        self.cv1 = ConvBNSiLU(in_c, hidden, 1)
        self.cv2 = ConvBNSiLU(in_c, hidden, 1)
        self.cv3 = ConvBNSiLU(2 * hidden, out_c, 1)
        self.blocks = nn.Sequential(*[
            ConvBNSiLU(hidden, hidden, 3, p=1) for _ in range(num_blocks)
        ])

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.blocks(self.cv2(x))
        return self.cv3(torch.cat([y1, y2], dim=1))


class CSPDarknet(nn.Module):
    """Light CSPDarknet — multi-scale feature extractor."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = ConvBNSiLU(in_channels, 32, 3, s=1, p=1)

        # Stage 1:  → /2  (img/2)
        self.s1_down = ConvBNSiLU(32, 64, 3, s=2, p=1)
        self.s1_csp  = CSPLayer(64, 64, num_blocks=2)

        # Stage 2:  → /4  (img/4) = P3
        self.s2_down = ConvBNSiLU(64, 128, 3, s=2, p=1)
        self.s2_csp  = CSPLayer(128, 128, num_blocks=4)

        # Stage 3:  → /8  (img/8) = P4
        self.s3_down = ConvBNSiLU(128, 256, 3, s=2, p=1)
        self.s3_csp  = CSPLayer(256, 256, num_blocks=6)

        # Stage 4:  → /16 (img/16) = P5 (last scale — spatial already at 14×14 for 224)
        self.s4_down = ConvBNSiLU(256, 512, 3, s=2, p=1)
        self.s4_csp  = CSPLayer(512, 512, num_blocks=4)

    def forward(self, x):
        x   = self.stem(x)
        x   = self.s1_down(x); x = self.s1_csp(x)    # /2  skip
        p3  = self.s2_down(x); p3 = self.s2_csp(p3)  # /4
        p4  = self.s3_down(p3); p4 = self.s3_csp(p4) # /8
        p5  = self.s4_down(p4); p5 = self.s4_csp(p5) # /16
        return p3, p4, p5  # [C:128, C:256, C:512]


# ============================================================================
# 2. FPN + PAN neck  (YOLO-style multi-scale fusion)
# ============================================================================

class FPNPAN(nn.Module):
    """FPN (top-down) + PAN (bottom-up) — the heart of YOLO's detection neck."""

    def __init__(self, p3_c=128, p4_c=256, p5_c=512, out_c=256):
        super().__init__()
        # ── FPN top-down ──
        self.lat_p5 = ConvBNSiLU(p5_c, out_c, 1)
        self.lat_p4 = ConvBNSiLU(p4_c, out_c, 1)
        self.lat_p3 = ConvBNSiLU(p3_c, out_c, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # post-fuse (FPN)
        self.fuse_p4 = CSPLayer(2 * out_c, out_c, num_blocks=2)
        self.fuse_p3 = CSPLayer(2 * out_c, out_c, num_blocks=2)  # N3 = final 1/4

        # ── PAN bottom-up ──
        self.down_conv_p3 = ConvBNSiLU(out_c, out_c, 3, s=2, p=1)  # N3 → /8
        self.fuse_n4      = CSPLayer(2 * out_c, out_c, num_blocks=2)
        self.down_conv_p4 = ConvBNSiLU(out_c, out_c, 3, s=2, p=1)  # N4 → /16
        self.fuse_n5      = CSPLayer(2 * out_c, out_c, num_blocks=2)

    def forward(self, p3, p4, p5):
        # Top-down (FPN)
        p5_lat = self.lat_p5(p5)                          # [B, out_c, /16, /16]
        p4_lat = self.lat_p4(p4)                          # [B, out_c, /8,  /8]
        p3_lat = self.lat_p3(p3)                          # [B, out_c, /4,  /4]

        p4_up = self.upsample(p5_lat)                     # /16 → /8
        n4    = self.fuse_p4(torch.cat([p4_lat, p4_up], dim=1))

        p3_up = self.upsample(n4)                         # /8 → /4
        n3    = self.fuse_p3(torch.cat([p3_lat, p3_up], dim=1))

        # Bottom-up (PAN)
        n3_down = self.down_conv_p3(n3)                   # /4 → /8
        n4_out  = self.fuse_n4(torch.cat([n4, n3_down], dim=1))

        n4_down = self.down_conv_p4(n4_out)               # /8 → /16
        n5_out  = self.fuse_n5(torch.cat([p5_lat, n4_down], dim=1))

        return n3, n4_out, n5_out  # same scales as P3/P4/P5 but fused


# ============================================================================
# 3. SpatialConv Prediction Head  (grid-aware conv, not flat MLP)
# ============================================================================

class SpatialGridHead(nn.Module):
    """Conv head that preserves spatial structure, parallel to YOLO's detect head.

    Takes multi-scale neck outputs → grid of features per scale → concat.
    Inputs can come from one or two image streams (RGB + Elevation shared neck).
    """

    def __init__(self, in_c: int = 256, grid_dim: int = 256):
        super().__init__()
        # Reduce each scale to a uniform grid feature map
        self.head_p5 = nn.Conv2d(in_c, grid_dim, 1)   # /16
        self.head_p4 = nn.Conv2d(in_c, grid_dim, 1)   # /8
        self.head_p3 = nn.Conv2d(in_c, grid_dim, 1)   # /4

    def forward(self, n3, n4, n5, target_size: int = 7):
        """Extract and interpolate to uniform grid, then concat channel-wise.

        Returns [B, H, W, 3*grid_dim] — spatial grid at target_size.
        """
        f5 = F.interpolate(self.head_p5(n5), size=(target_size, target_size), mode='bilinear')
        f4 = F.interpolate(self.head_p4(n4), size=(target_size, target_size), mode='bilinear')
        f3 = F.interpolate(self.head_p3(n3), size=(target_size, target_size), mode='bilinear')
        fused = torch.cat([f3, f4, f5], dim=1)  # [B, 3*grid_dim, H, W]
        return fused.permute(0, 2, 3, 1)         # [B, H, W, C]


# ============================================================================
# 4. Position encoding (2D spatial + 1D temporal)
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, device=None):
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4")
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _1d_sincos(embed_dim // 2, grid[0].reshape(-1))  # [H*W, D/2]
    emb_w = _1d_sincos(embed_dim // 2, grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)[np.newaxis, :, :]  # [1, H*W, D]
    return torch.from_numpy(emb).float().to(device)


def _1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class SpatioTemporalPosEmbed(nn.Module):
    def __init__(self, max_frames: int, grid_size: int, grid_dim: int):
        super().__init__()
        self.temp_embed = nn.Parameter(torch.randn(1, max_frames, 1, 1, 1) * 0.02)
        spatial = get_2d_sincos_pos_embed(grid_dim, grid_size)  # [1, H*W, D]
        self.register_buffer("spatial_embed", spatial)

    def forward(self, features):
        B, T, H, W, D = features.shape
        pos_spatial = self.spatial_embed[:, : H * W, :].view(1, 1, H, W, D)
        pos_temp    = self.temp_embed[:, :T, :, :, :]
        return features + pos_spatial + pos_temp


# ============================================================================
# 5. Main model
# ============================================================================

class ExcavatorVLAYolo(nn.Module):
    """YOLO-style spatio-temporal model for excavator joint prediction.

    Input:
        rgb          [B, T, 3, H, W]
        elevation    [B, T, 3, H, W]
        qpos         [B, T, 4]     (optional, train-only aux)
        excavator_id [B]
    Output:
        sin/cos pairs [B, 8]   (decode with atan2 for rad)
    """

    def __init__(
        self, seq_len=8, img_size=224, hidden_dim=512,
        n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1,
        pretrained=True, num_excavators=4,
        use_sincos_output=True,
        qpos_mode="modulation",  # "none" | "modulation" | "transformer"
        qpos_drop_prob=0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.use_sincos_output = use_sincos_output
        self.qpos_mode = qpos_mode
        self.qpos_drop_prob = qpos_drop_prob
        out_dim = 8 if use_sincos_output else 4

        # ── Dual backbone + FPN-PAN neck ──
        neck_out = 256  # uniform channel after FPN-PAN per scale
        grid_dim = 256  # per-scale → grid_dim after head

        # Shared neck for both streams (weight-sharing saves memory)
        self.neck = FPNPAN(p3_c=128, p4_c=256, p5_c=512, out_c=neck_out)

        # RGB stream
        self.rgb_backbone = CSPDarknet(3)
        self.rgb_head = SpatialGridHead(neck_out, grid_dim)

        # Elevation stream
        self.elev_backbone = CSPDarknet(3)
        self.elev_head = SpatialGridHead(neck_out, grid_dim)

        # ── Fused grid dimension ──
        grid_size = img_size // 16  # using /16 scale (14×14 for 224)
        total_grid_dim = 3 * grid_dim * 2  # 3 scales × 2 streams
        self.grid_size = grid_size
        self.grid_proj = nn.Linear(total_grid_dim, hidden_dim)

        # ── Self-supervised Task-Region Mask Head ──
        # Learns K functional zones (loading/unloading/digging area, etc.)
        # No annotations! Self-supervised via:
        #   1. Contrast:  randomly masks OUT regions → prediction must still work
        #   2. Sparsity:   each mask focuses on a compact region
        #   3. Temporal:   masks are smooth over consecutive frames
        self.num_regions = 4  # functional task regions to discover
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_regions),
        )

        # ── Proprioception ──
        if qpos_mode == "modulation":
            self.qpos_mod = nn.Sequential(
                nn.Linear(4, 32), nn.GELU(), nn.Linear(32, out_dim),
            )
        elif qpos_mode == "transformer":
            self.qpos_proj = nn.Sequential(
                nn.Linear(4, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
        else:
            self.qpos_mod = None

        # ── Excavator ID embedding ──
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)

        # ── Position encoding ──
        self.pos_embed = SpatioTemporalPosEmbed(seq_len, grid_size, hidden_dim)

        # ── CLS token (now N query tokens for decoder) ──
        self.num_queries = 4
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim) * 0.02)

        # ── Transformer Encoder (deep, processes grid tokens) ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Transformer Decoder (cross-attends queries → encoder output) ──
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(2, n_layers // 2))

        # ── Readout head ──
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.5)
        for p in self.decoder.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.5)
        for module in self.delta_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None: nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)
        if self.qpos_mode == "modulation":
            for module in self.qpos_mod:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None: nn.init.zeros_(module.bias)

    def decode_delta(self, raw: torch.Tensor) -> torch.Tensor:
        if self.use_sincos_output:
            sin, cos = raw.chunk(2, dim=-1)
            return torch.atan2(sin, cos)
        return raw

    def forward(self, rgb, elevation, qpos=None, excavator_id=None):
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]
        G = self.grid_size
        D = self.hidden_dim

        # ── Per-frame CSPDarknet + FPN-PAN + SpatialHead ──
        rgb_flat  = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)

        # RGB stream
        p3_rgb, p4_rgb, p5_rgb = self.rgb_backbone(rgb_flat)
        n3_rgb, n4_rgb, n5_rgb = self.neck(p3_rgb, p4_rgb, p5_rgb)
        grid_rgb = self.rgb_head(n3_rgb, n4_rgb, n5_rgb, G)  # [B*T, G, G, 3*grid_dim]

        # Elevation stream
        p3_elev, p4_elev, p5_elev = self.elev_backbone(elev_flat)
        n3_elev, n4_elev, n5_elev = self.neck(p3_elev, p4_elev, p5_elev)
        grid_elev = self.elev_head(n3_elev, n4_elev, n5_elev, G)

        # Fuse streams at grid level
        grid = torch.cat([grid_rgb, grid_elev], dim=-1)   # [B*T, G, G, 6*grid_dim]
        grid = self.grid_proj(grid)                        # [B*T, G, G, D]
        grid = grid.view(B, T, G, G, D)

        # ── qpos injection (transformer mode, train only) ──
        if self.qpos_mode == "transformer" and qpos is not None and self.training:
            mask = torch.rand(B, 1, 1, 1, 1, device=qpos.device) > self.qpos_drop_prob \
                   if self.qpos_drop_prob > 0 else torch.ones(B, 1, 1, 1, 1, device=qpos.device)
            grid = grid + self.qpos_proj(qpos).unsqueeze(2).unsqueeze(2) * mask.float()

        # ── Position encoding ──
        grid = self.pos_embed(grid)

        # ── Tokens ──
        tokens = grid.reshape(B, T * G * G, D)
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # ── Encoder: process grid tokens ──
        memory = self.encoder(tokens)  # [B, N_tokens, D]

        # ── Task-region masks: learn functional zones per frame ──
        # memory [B, T×G×G, D] → K-channel activation → softmax over spatial dim
        raw_scores = self.mask_head(memory)              # [B, T×G×G, K]
        raw_scores = raw_scores.view(B, T, G * G, self.num_regions)
        raw_scores = raw_scores.permute(0, 3, 1, 2)      # [B, K, T, G×G]

        # Softmax over spatial positions → each region picks its cells
        masks = F.softmax(raw_scores, dim=-1)             # [B, K, T, G×G]
        masks = masks.view(B, self.num_regions, T, G, G)  # [B, K, T, G, G]

        # ── Decoder: N queries cross-attend to encoder memory ──
        queries = self.query_tokens.expand(B, -1, -1)           # [B, Q, D]
        decoded = self.decoder(queries, memory)                  # [B, Q, D]

        # ── Aggregate queries → delta ──
        pool = decoded.mean(dim=1)                               # [B, D]
        delta = self.delta_head(pool)

        # ── qpos modulation residual (train only) ──
        if self.qpos_mode == "modulation" and qpos is not None and self.training:
            qpos_last = qpos[:, -1, :]
            correction = self.qpos_mod(qpos_last)
            if self.qpos_drop_prob > 0:
                m_drop = (torch.rand(B, 1, device=qpos.device) > self.qpos_drop_prob).float()
                correction = correction * m_drop
            delta = delta + correction

        # Time-averaged masks for visualization
        avg_masks = masks.mean(dim=2)  # [B, K, G, G]

        return delta, avg_masks


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
