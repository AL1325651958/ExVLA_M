"""Excavator-STVTA V12: Dual-branch Spatio-Temporal Video-to-Action.

RGB and Elevation each have independent: backbone → neck → temporal mixer →
mask generator → decoder → joint features. The two branches only meet at
per-joint modality fusion. This prevents early cross-modal contamination and
allows testing whether each modality contributes usefully to each joint.

Naming: STVTA (Spatio-Temporal Video-to-Action), not YOLO.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse shared building blocks from the YOLO model
from vla_model.model_yolo import (
    ConvBNSiLU, CSPLayer, CSPDarknet, FPNPAN, SpatialGridHead,
    SpatioTemporalPosEmbed, TemporalMaskMixer,
    MaskBiasedCrossAttn, MaskBiasedDecoderLayer,
    get_2d_sincos_pos_embed, _1d_sincos,
    count_parameters as _count_params,
)


# ── V12: Single-branch encoder (backbone → neck → grid → temporal → masks → decoder) ──


class MotionEncoder(nn.Module):
    """Encode frame-difference video into grid-compatible features.
    Input: [B, T, 3, H, W] frame residuals (first frame zeroed).
    Output: [B, T, G, G, D] motion features that add to the visual grid.
    """
    def __init__(self, hidden_dim=512, grid_size=14):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
        )
        # Adaptive pooling to grid size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

    def forward(self, frame_diff):
        """frame_diff: [B*T, 3, H, W] -> motion: [B, T, G, G, D]"""
        BT = frame_diff.shape[0]
        x = self.encoder(frame_diff)            # [B*T, D, H/8, W/8]
        x = self.pool(x)                        # [B*T, D, G, G]
        x = x.permute(0, 2, 3, 1)              # [B*T, G, G, D]
        return x


class SingleModalityBranch(nn.Module):
    """One complete branch for RGB or Elevation.

    Input: video [B, T, 3, H, W]
    Output: joint_features [B, 4, D], masks_spatial [B, 4, T, G, G]
    """
    def __init__(self, name, hidden_dim=512, n_heads=8, n_layers=4, ff_dim=2048,
                 dropout=0.1, grid_dim=256, neck_out=256, grid_size=14,
                 input_adapter=None):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_joints = 4
        self.input_adapter = input_adapter

        # Motion encoder for frame-difference features
        self.motion_encoder = MotionEncoder(hidden_dim, grid_size)

        # Backbone + neck
        self.backbone = CSPDarknet(3)
        self.neck = FPNPAN(p3_c=128, p4_c=256, p5_c=512, out_c=neck_out)
        self.grid_head = SpatialGridHead(neck_out, grid_dim)

        # Grid projection (separate per branch)
        grid_dim_total = 3 * grid_dim  # p3+p4+p5 × 256dim
        self.grid_proj = nn.Linear(grid_dim_total, hidden_dim)

        # Temporal mixer
        self.temporal_mixer = TemporalMaskMixer(
            hidden_dim, nhead=max(4, n_heads // 2), num_layers=1,
            ff_dim=hidden_dim * 2, dropout=dropout,
        )

        # Joint-conditioned mask generator (shared MLP, conditioned by joint_embed)
        self.joint_embed = nn.Parameter(torch.randn(self.num_joints, hidden_dim) * 0.02)
        self.mask_generator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Position + excavator embedding
        self.pos_embed = SpatioTemporalPosEmbed(8, grid_size, hidden_dim)

        # Encoder (shared across joints within this modality)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder layers (mask-biased cross-attention)
        self.decoder_layers = nn.ModuleList([
            MaskBiasedDecoderLayer(hidden_dim, n_heads, ff_dim, dropout, lambda_mask=3.0)
            for _ in range(max(2, n_layers // 2))
        ])

        # Joint queries
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, hidden_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        nn.init.normal_(self.joint_embed, mean=0.0, std=0.02)
        for module in self.mask_generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.mask_generator[-1].bias, -1.0)

    def forward(self, video, excavator_id=None):
        """video: [B, T, 3, H, W] → joint_features [B, 4, D], masks [B, 4, T, G, G]"""
        B, T, _, H, W = video.shape
        G = H // 16   # grid size (224/16=14)
        D = self.hidden_dim

        # Frame-difference features (V12.2)
        x = video.reshape(B * T, 3, H, W)
        # Compute frame residuals: [f0, f1, f2, ...] -> [0, f1-f0, f2-f1, ...]
        frame_diff = torch.zeros_like(x)
        frame_diff[B:] = x[B:] - x[:-B]  # frame_diff[t] = x[t] - x[t-B]
        motion = self.motion_encoder(frame_diff).view(B, T, G, G, D)

        # Vision backbone
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        grid = self.grid_head(n3, n4, n5, G)                     # [B*T, G, G, 3*256]

        grid = self.grid_proj(grid).view(B, T, G, G, D)

        # Add motion features before temporal mixing
        grid = grid + motion * 0.1

        # Temporal mixing
        grid = self.temporal_mixer(grid)

        # Position + excavator encoding
        grid = self.pos_embed(grid)
        tokens = grid.reshape(B, T * G * G, D)
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # Joint-conditioned masks
        masks_list = []
        for j in range(self.num_joints):
            cond = tokens + self.joint_embed[j]
            m_j = torch.sigmoid(self.mask_generator(cond)).squeeze(-1)  # [B, N]
            masks_list.append(m_j)
        masks_flat = torch.stack(masks_list, dim=1)               # [B, 4, N]
        masks_spatial = masks_flat.view(B, self.num_joints, T, G, G)

        # Soft union gate for encoder
        gate = 1.0 - (1.0 - masks_flat).prod(dim=1)
        gate = 0.02 + 0.98 * gate
        gated_tokens = tokens * gate.unsqueeze(-1)

        # Encoder
        memory = self.encoder(gated_tokens)                        # [B, N, D]

        # Mask-biased joint decoder
        decoded_list = []
        for j in range(self.num_joints):
            tgt = self.joint_queries[:, j:j+1, :].expand(B, -1, -1)
            for layer in self.decoder_layers:
                tgt = layer(tgt, memory, masks_flat[:, j, :])
            decoded_list.append(tgt)
        decoded = torch.cat(decoded_list, dim=1)                   # [B, 4, D]

        return decoded, masks_spatial


# ── V12 Main Model ──

class ExcavatorSTVTA(nn.Module):
    """Excavator-STVTA V12: Dual-branch spatio-temporal video-to-action.

    Two isolated branches (RGB, Elevation) each produce joint features and
    masks. Per-joint fusion gates mix the two branches before action heads.
    """
    def __init__(
        self, seq_len=8, img_size=224, hidden_dim=512,
        n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1,
        pretrained=True, num_excavators=4, version="v12",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.num_joints = 4
        self.version = version

        self.out_dims = [2, 2, 2, 2]
        self.out_dim = 8
        self.num_excavators = num_excavators

        G = img_size // 16  # grid size

        # ── Elevation modality adapter ──
        elev_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

        # ── Two independent branches ──
        self.rgb_branch = SingleModalityBranch(
            "rgb", hidden_dim, n_heads, n_layers, ff_dim, dropout,
            grid_size=G, input_adapter=None,
        )
        self.elev_branch = SingleModalityBranch(
            "elev", hidden_dim, n_heads, n_layers, ff_dim, dropout,
            grid_size=G, input_adapter=elev_adapter,
        )

        # ── Shared excavator embedding (used by both branches) ──
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)
        self.rgb_branch.excv_embed = self.excv_embed
        self.elev_branch.excv_embed = self.excv_embed

        # ── Per-joint fusion gates ──
        # alpha_j = sigmoid(W·[rgb_joint_j; elev_joint_j])
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
            ) for _ in range(self.num_joints)
        ])

        # ── Per-excavator per-joint action heads ──
        self.action_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
                    nn.Linear(128, 2),  # sin/cos per joint
                ) for _ in range(self.num_joints)
            ]) for _ in range(num_excavators)
        ])

        # ── Training-only pose auxiliary ──
        self.pose_aux_head = nn.Linear(hidden_dim * 2, 4)

        self._init_weights()

    def _init_weights(self):
        for excv_heads in self.action_heads:
            for head in excv_heads:
                for module in head:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)


    def decode_action(self, raw):
        """raw [B, 8] → [B, 4] rad, projected onto unit circle."""
        raw_4d = raw.view(-1, self.num_joints, 2)
        norm = raw_4d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        raw_4d = raw_4d / norm
        return torch.atan2(raw_4d[..., 0], raw_4d[..., 1])

    def forward(self, rgb, elevation, qpos=None, excavator_id=None,
                return_aux=False, return_diagnostics=False):
        """Pure visual inference. qpos is unused; it exists only for API compat.

        Args:
            return_aux: add pose_aux [B,4] (training only)
            return_diagnostics: add fusion_alpha [B,4] for visualization

        Returns:
            action [B,8], avg_masks [B,2,4,G,G], spatial_masks [B,2,4,T,G,G]
            (+ pose_aux if return_aux, + fusion_alpha if return_diagnostics)
        """
        B, T, _, H, W = rgb.shape
        G = H // 16
        training = self.training

        # Modality dropout disabled in V12.2 — establish clean dual-modal baseline

        # ── RGB branch ──
        rgb_features, rgb_masks = self.rgb_branch(rgb, excavator_id)

        # ── Elevation branch (adapter is inside the branch now) ──
        elev_features, elev_masks = self.elev_branch(elevation, excavator_id)

        # ── Stack masks: [B, 2=RGB/Elev, 4, T, G, G] ──
        masks_spatial = torch.stack([rgb_masks, elev_masks], dim=1)
        avg_masks = masks_spatial.mean(dim=3)                     # [B, 2, 4, G, G]

        # ── Mask statistics (for monitoring mask diversity) ──
        # Spatial std per mask: how concentrated is each mask spatially?
        mask_spatial_std = masks_spatial.std(dim=(-2, -1)).mean()  # scalar
        # Temporal std per mask: how much do masks change across frames?
        mask_temporal_std = (masks_spatial[:, :, :, 1:] -
                             masks_spatial[:, :, :, :-1]).abs().mean()  # scalar

        # ── Per-joint fusion ──
        fused_list = []
        alpha_list = []
        for j in range(self.num_joints):
            cat_j = torch.cat([rgb_features[:, j], elev_features[:, j]], dim=-1)  # [B, 2D]
            alpha_j = self.fusion_gates[j](cat_j)                 # [B, 1]
            fused_j = alpha_j * rgb_features[:, j] + (1.0 - alpha_j) * elev_features[:, j]
            fused_list.append(fused_j)
            alpha_list.append(alpha_j)
        fused = torch.stack(fused_list, dim=1)                    # [B, 4, D]
        fusion_alpha = torch.cat(alpha_list, dim=-1)              # [B, 4]

        # ── Per-excavator per-joint action heads ──
        action = torch.zeros(B, self.out_dim, device=fused.device, dtype=fused.dtype)
        for eid in range(self.num_excavators):
            mask_e = (excavator_id == eid)
            if mask_e.any():
                acts_e = []
                for j in range(self.num_joints):
                    acts_e.append(self.action_heads[eid][j](fused[mask_e, j]))
                action[mask_e] = torch.cat(acts_e, dim=-1).float()

        # ── Optional outputs ──
        outputs = (action, avg_masks, masks_spatial)
        if return_aux:
            # Pool last-frame features from both branches for current-pose prediction
            rgb_pool = rgb_features.mean(dim=1)  # D
            elev_pool = elev_features.mean(dim=1)
            pose_aux = self.pose_aux_head(torch.cat([rgb_pool, elev_pool], dim=-1))
            outputs = (*outputs, pose_aux)
        if return_diagnostics:
            mask_stats = {"spatial_std": mask_spatial_std.item(),
                          "temporal_std": mask_temporal_std.item()}
            outputs = (*outputs, fusion_alpha, mask_stats)
            return outputs
        outputs = (*outputs, fusion_alpha)
        return outputs


def count_parameters(model):
    return _count_params(model)
