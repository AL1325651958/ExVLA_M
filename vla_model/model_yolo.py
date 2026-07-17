import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def upgrade_legacy_v17_1_state_dict(state_dict):
    """Map legacy shared V17.1 mask heads onto four independent heads."""
    upgraded = dict(state_dict)
    if "mask_linear1.weight" not in upgraded or "mask_linear1.0.weight" in upgraded:
        return upgraded

    for joint in range(4):
        source_prefix = "swing_mask_linear" if joint == 3 else "mask_linear"
        for layer in (1, 2):
            for parameter in ("weight", "bias"):
                source = f"{source_prefix}{layer}.{parameter}"
                target = f"mask_linear{layer}.{joint}.{parameter}"
                if source in upgraded:
                    upgraded[target] = upgraded[source]
    return upgraded


def load_compatible_state_dict(module, state_dict):
    """Load checkpoint tensors whose names and shapes match ``module``."""
    module_state = module.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in module_state and module_state[key].shape == value.shape
    }
    module.load_state_dict(compatible, strict=False)
    return len(compatible), len(state_dict) - len(compatible)


class ConvBNSiLU(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPLayer(nn.Module):
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
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = ConvBNSiLU(in_channels, 32, 3, s=1, p=1)
        self.s1_down = ConvBNSiLU(32, 64, 3, s=2, p=1)
        self.s1_csp = CSPLayer(64, 64, num_blocks=2)
        self.s2_down = ConvBNSiLU(64, 128, 3, s=2, p=1)
        self.s2_csp = CSPLayer(128, 128, num_blocks=4)
        self.s3_down = ConvBNSiLU(128, 256, 3, s=2, p=1)
        self.s3_csp = CSPLayer(256, 256, num_blocks=6)
        self.s4_down = ConvBNSiLU(256, 512, 3, s=2, p=1)
        self.s4_csp = CSPLayer(512, 512, num_blocks=4)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1_down(x)
        x = self.s1_csp(x)
        p3 = self.s2_down(x)
        p3 = self.s2_csp(p3)
        p4 = self.s3_down(p3)
        p4 = self.s3_csp(p4)
        p5 = self.s4_down(p4)
        p5 = self.s4_csp(p5)
        return p3, p4, p5


class FPNPAN(nn.Module):
    def __init__(self, p3_c=128, p4_c=256, p5_c=512, out_c=256):
        super().__init__()
        self.lat_p5 = ConvBNSiLU(p5_c, out_c, 1)
        self.lat_p4 = ConvBNSiLU(p4_c, out_c, 1)
        self.lat_p3 = ConvBNSiLU(p3_c, out_c, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_p4 = CSPLayer(2 * out_c, out_c, num_blocks=2)
        self.fuse_p3 = CSPLayer(2 * out_c, out_c, num_blocks=2)
        self.down_conv_p3 = ConvBNSiLU(out_c, out_c, 3, s=2, p=1)
        self.fuse_n4 = CSPLayer(2 * out_c, out_c, num_blocks=2)
        self.down_conv_p4 = ConvBNSiLU(out_c, out_c, 3, s=2, p=1)
        self.fuse_n5 = CSPLayer(2 * out_c, out_c, num_blocks=2)

    def forward(self, p3, p4, p5):
        p5_lat = self.lat_p5(p5)
        p4_lat = self.lat_p4(p4)
        p3_lat = self.lat_p3(p3)
        n4 = self.fuse_p4(torch.cat([p4_lat, self.upsample(p5_lat)], dim=1))
        n3 = self.fuse_p3(torch.cat([p3_lat, self.upsample(n4)], dim=1))
        n3_down = self.down_conv_p3(n3)
        n4_out = self.fuse_n4(torch.cat([n4, n3_down], dim=1))
        n4_down = self.down_conv_p4(n4_out)
        n5_out = self.fuse_n5(torch.cat([p5_lat, n4_down], dim=1))
        return n3, n4_out, n5_out


class SpatialGridHead(nn.Module):
    def __init__(self, in_c=256, grid_dim=256):
        super().__init__()
        self.head_p5 = nn.Conv2d(in_c, grid_dim, 1)
        self.head_p4 = nn.Conv2d(in_c, grid_dim, 1)
        self.head_p3 = nn.Conv2d(in_c, grid_dim, 1)

    def forward(self, n3, n4, n5, target_size=7):
        f5 = F.interpolate(self.head_p5(n5), size=(target_size, target_size), mode='bilinear')
        f4 = F.interpolate(self.head_p4(n4), size=(target_size, target_size), mode='bilinear')
        f3 = F.interpolate(self.head_p3(n3), size=(target_size, target_size), mode='bilinear')
        return torch.cat([f3, f4, f5], dim=1).permute(0, 2, 3, 1)


def _1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, device=None):
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _1d_sincos(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _1d_sincos(embed_dim // 2, grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)[np.newaxis, :, :]
    return torch.from_numpy(emb).float().to(device)


class SpatioTemporalPosEmbed(nn.Module):
    def __init__(self, max_frames, grid_size, grid_dim):
        super().__init__()
        self.temp_embed = nn.Parameter(torch.randn(1, max_frames, 1, 1, 1) * 0.02)
        spatial = get_2d_sincos_pos_embed(grid_dim, grid_size)
        self.register_buffer("spatial_embed", spatial)

    def forward(self, features):
        B, T, H, W, D = features.shape
        pos_s = self.spatial_embed[:, : H * W, :].view(1, 1, H, W, D)
        pos_t = self.temp_embed[:, :T, :, :, :]
        return features + pos_s + pos_t


class MaskBiasedCrossAttn(nn.Module):
    """Cross-attention where mask_j controls BOTH attention logits AND value.

    A_ij = softmax_j(Q_i·K_j/√d + λ_log·log(M_j+ε))
    out_i = Σ_j A_ij · (V_j · (1 + λ_v · M_j))

    Residual value gating: mask≈0 keeps base signal; mask≈1 boosts by factor (1+λ_v).
    No information is zeroed out — masks only enhance relevant regions.

    Per-call overrides: use_logit_bias=False disables logit bias (for Swing);
    lambda_value overrides the module default (for Swing's weak residual).
    """
    def __init__(self, d_model, nhead, dropout=0.1, lambda_mask=3.0, lambda_value=1.5, residual=True):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.lambda_mask = lambda_mask
        self.lambda_value = lambda_value
        self.residual = residual
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, mask_j, use_logit_bias=True, lambda_mask=None, lambda_value=None):
        B, N, D = memory.shape
        Q = self.q_proj(query).view(B, 1, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(memory).view(B, N, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(memory).view(B, N, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        attn = Q @ K.transpose(-2, -1) * self.scale
        if use_logit_bias:
            lm = lambda_mask if lambda_mask is not None else self.lambda_mask
            log_m = (mask_j + 1e-6).log().unsqueeze(1).unsqueeze(2)
            attn = attn + lm * log_m
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        m = mask_j.unsqueeze(1).unsqueeze(-1)                      # [B,1,1,N]
        lv = lambda_value if lambda_value is not None else self.lambda_value
        if self.residual:
            V_gated = V * (1.0 + lv * m)                            # residual boost, no zeroing
        else:
            V_gated = V * m
        out = (attn @ V_gated).permute(0, 2, 1, 3).reshape(B, 1, D)
        return self.out_proj(out)


class MaskBiasedDecoderLayer(nn.Module):
    """Decoder layer: self-attn → mask-biased cross-attn → FFN (norm_first)."""
    def __init__(self, d_model, nhead, ff_dim, dropout=0.1, lambda_mask=3.0, lambda_value=1.5, residual=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MaskBiasedCrossAttn(d_model, nhead, dropout, lambda_mask, lambda_value, residual)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, tgt, memory, mask_j, use_logit_bias=True, lambda_mask=None, lambda_value=None):
        tgt = tgt + self.dropout(self.self_attn(self.n1(tgt), self.n1(tgt), self.n1(tgt), need_weights=False)[0])
        tgt = tgt + self.cross_attn(self.n2(tgt), memory, mask_j, use_logit_bias=use_logit_bias, lambda_mask=lambda_mask, lambda_value=lambda_value)
        tgt = tgt + self.dropout(self.ff2(self.dropout(self.act(self.ff1(self.n3(tgt))))))
        return tgt


class TemporalMaskMixer(nn.Module):
    """Cross-frame temporal mixing at each spatial grid location.

    Reshapes [B, T, G, G, D] → [B*G*G, T, D], applies a Transformer encoder
    across the time axis at each location, then restores the original shape.

    This gives every spatial position a dedicated temporal context before
    mask generation — enabling masks that evolve with motion across frames.
    """
    def __init__(self, d_model, nhead=4, num_layers=1, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, grid):
        B, T, G, Gg, D = grid.shape
        x = grid.permute(0, 2, 3, 1, 4).reshape(B * G * Gg, T, D)
        x = self.encoder(x)
        x = x.reshape(B, G, Gg, T, D).permute(0, 3, 1, 2, 4)
        return x


class ExcavatorVLAYolo(nn.Module):
    def __init__(
        self, seq_len=8, img_size=224, hidden_dim=512,
        n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1,
        pretrained=True, num_excavators=4, version="v9",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.num_joints = 4
        self.version = version

        # Mixed output: all sin/cos — stable bounded representation
        self.out_dims = [2, 2, 2, 2]       # per-joint sin/cos
        self.out_dim = sum(self.out_dims)  # 8

        neck_out = 256
        grid_dim = 256
        grid_size = img_size // 16
        self.grid_size = grid_size

        # ── Vision encoders ──
        self.rgb_backbone = CSPDarknet(3)
        self.rgb_neck = FPNPAN(p3_c=128, p4_c=256, p5_c=512, out_c=neck_out)
        self.rgb_head = SpatialGridHead(neck_out, grid_dim)

        # Elevation modality adapter: lightweight stem converts domain
        self.elev_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        self.elev_backbone = CSPDarknet(3)
        self.elev_neck = FPNPAN(p3_c=128, p4_c=256, p5_c=512, out_c=neck_out)
        self.elev_head = SpatialGridHead(neck_out, grid_dim)

        # ── Grid projection ──
        grid_size = img_size // 16
        total_grid_dim = 3 * grid_dim * 2
        self.grid_size = grid_size
        self.grid_proj = nn.Linear(2 * hidden_dim, hidden_dim)  # V16: cat of 2×D → D

        # V16: separate per-modality projections + cross-modal attention
        self.rgb_proj = nn.Linear(grid_dim * 3, hidden_dim)       # 768 → 512
        self.elev_proj = nn.Linear(grid_dim * 3, hidden_dim)      # 768 → 512
        self.cross_rgb_from_elev = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_elev_from_rgb = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)

        # ── V10+ temporal mask mixer (across-frame at each grid location) ──
        if version in ("v10", "v11", "v17.1"):
            self.temporal_mask_mixer = TemporalMaskMixer(
                hidden_dim, nhead=max(4, n_heads // 2), num_layers=1,
                ff_dim=hidden_dim * 2, dropout=dropout,
            )
            # V17.1 uses periodic pose supervision so Swing never sees a
            # discontinuity between -pi and +pi.  Earlier versions retain
            # their original four-radian auxiliary interface.
            pose_aux_dim = 8 if version == "v17.1" else 4
            self.pose_aux_head = nn.Linear(hidden_dim, pose_aux_dim)
        else:
            self.temporal_mask_mixer = None
            self.pose_aux_head = None

        # Independent training-only prediction of the next Swing increment.
        # It consumes visual decoder features, never qpos, and therefore does
        # not change the pure-visual inference contract.
        if version == "v17.1":
            self.swing_velocity_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            )
        else:
            self.swing_velocity_head = None

        # ── V11 pure-visual motion observation branch ──
        # Adjacent RGB/elevation frame residuals are encoded independently of
        # qpos, then softly fused into visual tokens before temporal mixing.
        if version == "v11":
            self.motion_adapter = nn.Sequential(
                nn.Conv2d(6, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.SiLU(),
                nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1),
            )
            self.motion_proj = nn.Linear(hidden_dim, hidden_dim)
            self.motion_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            )
        else:
            self.motion_adapter = None
            self.motion_proj = None
            self.motion_gate = None

        # ── Joint-conditioned mask heads ──
        # Each joint has a learnable embedding; mask_j sees tokens + joint_embed[j].
        # This guarantees mask_0 → Boom, mask_1 → Arm, mask_2 → Bucket, mask_3 → Swing.
        self.joint_embed = nn.Parameter(torch.randn(self.num_joints, hidden_dim) * 0.5)

        self.use_independent_joint_masks = version in ("v17", "v17.1")
        if self.use_independent_joint_masks:
            # V17: 4 independent mask heads — one per joint
            # Boom(0): semi-soft mask (needs boom + body + partial global)
            # Arm(1):  local mask (arm linkage region)
            # Bucket(2): strong local mask (bucket + end-effector zone)
            # Swing(3): global soft mask (full machine orientation + scene layout)
            self.mask_linear1 = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim // 2) for _ in range(4)
            ])
            self.mask_linear2 = nn.ModuleList([
                nn.Linear(hidden_dim // 2, 1) for _ in range(4)
            ])
            # Per-joint decoder parameters (registered as buffers so they survive .to(device))
            self.register_buffer('joint_logit_bias',
                torch.tensor([1.5, 2.0, 2.5, 0.0]))   # λ_mask: Boom<Arm<Bucket; Swing=0
            self.register_buffer('joint_value_lambda',
                torch.tensor([0.8, 1.0, 1.0, 0.5]))    # λ_v: Swing weakest, Boom medium
            # NOTE: Non-V17 paths still use the shared/swing_split mask heads below
            self.swing_mask_linear1 = None
            self.swing_mask_linear2 = None
        else:
            # V16 and earlier: shared head for Boom/Arm/Bucket + independent Swing head
            self.mask_linear1 = nn.Linear(hidden_dim, hidden_dim // 2)   # Boom/Arm/Bucket
            self.mask_linear2 = nn.Linear(hidden_dim // 2, 1)
            self.swing_mask_linear1 = nn.Linear(hidden_dim, hidden_dim // 2)  # Swing only
            self.swing_mask_linear2 = nn.Linear(hidden_dim // 2, 1)

        self.num_excavators = num_excavators
        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)
        self.pos_embed = SpatioTemporalPosEmbed(seq_len, grid_size, hidden_dim)

        # ── Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.decoder_layers = nn.ModuleList([
            MaskBiasedDecoderLayer(hidden_dim, n_heads, ff_dim, dropout, lambda_mask=3.0)
            for _ in range(max(2, n_layers // 2))
        ])

        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, hidden_dim) * 0.02)

        # Per-excavator per-joint action heads (different output dim per joint)
        self.action_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
                    nn.Linear(128, self.out_dims[j]),
                ) for j in range(self.num_joints)
            ]) for _ in range(num_excavators)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        for excv_heads in self.action_heads:
            for head in excv_heads:
                for module in head:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.joint_embed, mean=0.0, std=0.5)
        if self.use_independent_joint_masks:
            for j in range(4):
                nn.init.xavier_uniform_(self.mask_linear1[j].weight, gain=1.0)
                nn.init.zeros_(self.mask_linear1[j].bias)
                nn.init.xavier_uniform_(self.mask_linear2[j].weight, gain=0.5)
                nn.init.constant_(self.mask_linear2[j].bias, -1.0)
        else:
            nn.init.xavier_uniform_(self.mask_linear1.weight, gain=1.0)
            nn.init.zeros_(self.mask_linear1.bias)
            nn.init.xavier_uniform_(self.mask_linear2.weight, gain=0.5)
            nn.init.constant_(self.mask_linear2.bias, -1.0)
            nn.init.xavier_uniform_(self.swing_mask_linear1.weight, gain=1.0)
            nn.init.zeros_(self.swing_mask_linear1.bias)
            nn.init.xavier_uniform_(self.swing_mask_linear2.weight, gain=0.5)
            nn.init.constant_(self.swing_mask_linear2.bias, -1.0)
        # Elevation adapter: small init
        for module in self.elev_adapter:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.motion_adapter is not None:
            for module in self.motion_adapter:
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(self.motion_proj.weight, gain=0.1)
            nn.init.zeros_(self.motion_proj.bias)
            nn.init.xavier_uniform_(self.motion_gate[1].weight, gain=0.1)
            nn.init.zeros_(self.motion_gate[1].bias)

    @staticmethod
    def frame_residual(frames):
        """Return adjacent-frame visual residuals with a zero first timestep.

        ``frames`` has shape [B, T, C, H, W].  This is an observation-only
        operation used by V11; it never consumes joint state or actions.
        """
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, C, H, W]")
        residual = torch.zeros_like(frames)
        residual[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return residual

    def fuse_motion(self, visual_grid, motion):
        """Fuse motion evidence with a gate derived from visual grid tokens."""
        return visual_grid + self.motion_gate(visual_grid) * motion

    def decode_action(self, raw):
        """raw [B, 8] → [B, 4] rad, projected onto unit circle first."""
        raw_4d = raw.view(-1, self.num_joints, 2)
        norm = raw_4d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        raw_4d = raw_4d / norm
        return torch.atan2(raw_4d[..., 0], raw_4d[..., 1])

    def forward(self, rgb, elevation, qpos=None, excavator_id=None, return_aux=False):
        # qpos is deliberately unused: inference remains vision-only.
        # return_aux=True adds training-only auxiliary predictions. V10/V11
        # return pose_aux [B,4]; V17.1 returns periodic pose_aux [B,8] and
        # independent Swing velocity sin/cos [B,2].
        B, T = rgb.shape[:2]
        G = self.grid_size
        D = self.hidden_dim
        H, W = rgb.shape[3], rgb.shape[4]

        # ── RGB branch (independent neck) ──
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        p3_r, p4_r, p5_r = self.rgb_backbone(rgb_flat)
        n3_r, n4_r, n5_r = self.rgb_neck(p3_r, p4_r, p5_r)
        grid_rgb = self.rgb_head(n3_r, n4_r, n5_r, G)

        # ── Elevation branch (adapter + independent neck) ──
        elev_flat = elevation.reshape(B * T, 3, H, W)
        elev_adapted = self.elev_adapter(elev_flat)
        p3_e, p4_e, p5_e = self.elev_backbone(elev_adapted)
        n3_e, n4_e, n5_e = self.elev_neck(p3_e, p4_e, p5_e)
        grid_elev = self.elev_head(n3_e, n4_e, n5_e, G)

        # V16: project each modality separately, then cross-modal attention
        grid_rgb_flat = grid_rgb.view(B * T, G * G, -1)             # [BT, G², 768]
        grid_elev_flat = grid_elev.view(B * T, G * G, -1)           # [BT, G², 768]
        rgb_D = self.rgb_proj(grid_rgb_flat).view(B, T, G, G, D)   # [B, T, G, G, 512]
        elev_D = self.elev_proj(grid_elev_flat).view(B, T, G, G, D)

        # Cross-modal: RGB tokens attend to Elevation tokens and vice versa
        # Done as token-level exchange at each spatial position
        rgb_tokens = rgb_D.reshape(B, T * G * G, D)
        elev_tokens = elev_D.reshape(B, T * G * G, D)
        rgb_enhanced = self.cross_rgb_from_elev(rgb_tokens, elev_tokens, elev_tokens)[0]
        elev_enhanced = self.cross_elev_from_rgb(elev_tokens, rgb_tokens, rgb_tokens)[0]
        rgb_D = rgb_enhanced.reshape(B, T, G, G, D)
        elev_D = elev_enhanced.reshape(B, T, G, G, D)

        # V17.1 masks must observe motion across the input window.  Apply the
        # shared temporal mixer independently to both modalities before mask
        # generation; the fused action path then inherits the same features.
        if self.version == "v17.1" and self.temporal_mask_mixer is not None:
            rgb_D = self.temporal_mask_mixer(rgb_D)
            elev_D = self.temporal_mask_mixer(elev_D)

        # ── V16/V17: Independent masks from each modality before fusion ──
        # V17: each joint has its own mask_linear1[j]/mask_linear2[j] (ModuleList)
        # V16: Boom/Arm/Bucket share mask_linear1/2; Swing uses swing_mask_linear1/2
        rgb_t = rgb_D.reshape(B, T * G * G, D)
        elev_t = elev_D.reshape(B, T * G * G, D)
        rgb_masks_list, elev_masks_list = [], []
        for j in range(self.num_joints):
            if self.use_independent_joint_masks:
                # Independent mask head per joint
                h_r = F.gelu(self.mask_linear1[j](rgb_t + self.joint_embed[j]))
                m_r = torch.sigmoid(self.mask_linear2[j](h_r)).squeeze(-1)
                h_e = F.gelu(self.mask_linear1[j](elev_t + self.joint_embed[j]))
                m_e = torch.sigmoid(self.mask_linear2[j](h_e)).squeeze(-1)
            elif j == 3:
                # V16 Swing: independent head
                h_r = F.gelu(self.swing_mask_linear1(rgb_t + self.joint_embed[j]))
                m_r = torch.sigmoid(self.swing_mask_linear2(h_r)).squeeze(-1)
                h_e = F.gelu(self.swing_mask_linear1(elev_t + self.joint_embed[j]))
                m_e = torch.sigmoid(self.swing_mask_linear2(h_e)).squeeze(-1)
            else:
                # V16 Boom/Arm/Bucket: shared head
                h_r = F.gelu(self.mask_linear1(rgb_t + self.joint_embed[j]))
                m_r = torch.sigmoid(self.mask_linear2(h_r)).squeeze(-1)
                h_e = F.gelu(self.mask_linear1(elev_t + self.joint_embed[j]))
                m_e = torch.sigmoid(self.mask_linear2(h_e)).squeeze(-1)
            rgb_masks_list.append(m_r.view(B, T, G, G))
            elev_masks_list.append(m_e.view(B, T, G, G))
        rgb_masks = torch.stack(rgb_masks_list, dim=1)               # [B, 4, T, G, G]
        elev_masks = torch.stack(elev_masks_list, dim=1)             # [B, 4, T, G, G]

        masks_spatial = torch.stack([rgb_masks, elev_masks], dim=1)  # [B, 2, 4, T, G, G]

        # Fuse: RGB + Elev channels concatenated, then projected back to D
        grid = torch.cat([rgb_D, elev_D], dim=-1)
        grid = self.grid_proj(grid).view(B, T, G, G, D)

        # ── V11 residual motion fusion (pure visual; before temporal mixer) ──
        if self.motion_adapter is not None:
            rgb_delta = self.frame_residual(rgb)
            elev_delta = self.frame_residual(elevation)
            motion_input = torch.cat([rgb_delta, elev_delta], dim=2).reshape(B * T, 6, H, W)
            motion = self.motion_adapter(motion_input)
            motion = F.adaptive_avg_pool2d(motion, (G, G)).permute(0, 2, 3, 1)
            motion = self.motion_proj(motion).view(B, T, G, G, D)
            grid = self.fuse_motion(grid, motion)

        # ── V10+ temporal mask mixing (cross-frame at each grid location) ──
        if self.temporal_mask_mixer is not None and self.version != "v17.1":
            grid = self.temporal_mask_mixer(grid)

        grid = self.pos_embed(grid)
        tokens = grid.reshape(B, T * G * G, D)
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # ── Bidirectional history encoder ──
        # Window of 8 past frames are all observed; no causal mask needed.
        # NO encoder gating: global context is essential for Swing (rotation detection).
        # Mask-based filtering is applied only in the per-joint decoder cross-attention.
        memory = self.encoder(tokens)

        # ── Mask-biased joint cross-attention decoder ──
        # V17: per-joint graded parameters (Boom: λ_m=1.5/λ_v=0.8, Arm: 2.0/1.0,
        #   Bucket: 2.5/1.0, Swing: 0.0/0.5). All joints use residual gating only.
        # V16: Boom/Arm/Bucket use logit bias + residual; Swing uses weak residual only.
        decoded_list = []
        for j in range(self.num_joints):
            # Union of RGB + Elevation masks for this joint
            m_j = 1.0 - (1.0 - masks_spatial[:, 0, j]) * (1.0 - masks_spatial[:, 1, j])
            m_j = m_j.reshape(B, T * G * G)                            # [B, N]
            tgt = self.joint_queries[:, j:j+1, :].expand(B, -1, -1)
            if self.use_independent_joint_masks:
                lb = self.joint_logit_bias[j].item()
                lv = self.joint_value_lambda[j].item()
                use_bias = (lb > 0.0)
                for layer in self.decoder_layers:
                    tgt = layer(tgt, memory, m_j, use_logit_bias=use_bias,
                                lambda_mask=lb, lambda_value=lv)
            else:
                use_bias = (j != 3)                                    # Swing: no logit bias
                lv = 0.5 if j == 3 else None                           # Swing: weak residual
                for layer in self.decoder_layers:
                    tgt = layer(tgt, memory, m_j, use_logit_bias=use_bias, lambda_value=lv)
            decoded_list.append(tgt)
        decoded = torch.cat(decoded_list, dim=1)

        # ── Per-excavator per-joint action heads ──
        action = torch.zeros(B, self.out_dim, device=decoded.device, dtype=decoded.dtype)
        for eid in range(self.num_excavators):
            mask_e = (excavator_id == eid)
            if mask_e.any():
                acts_e = []
                for j in range(self.num_joints):
                    a_j = self.action_heads[eid][j](decoded[mask_e, j])  # [M, out_dims[j]]
                    acts_e.append(a_j)
                action[mask_e] = torch.cat(acts_e, dim=-1).float()

        avg_masks = masks_spatial.mean(dim=3)                       # [B, 2, 4, G, G]
        outputs = (action, avg_masks, masks_spatial)
        if return_aux and self.pose_aux_head is not None:
            # Pose auxiliary: predict current qpos from last-frame spatial tokens
            pose_aux = self.pose_aux_head(memory[:, -G * G:].mean(dim=1))
            if self.swing_velocity_head is not None:
                swing_velocity_aux = self.swing_velocity_head(decoded[:, 3])
                return (*outputs, pose_aux, swing_velocity_aux)
            return (*outputs, pose_aux)
        return outputs


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
