import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
    A_ij = softmax_j(Q_i·K_j/√d + λ·log(M_j+ε))
    out_i = Σ_j A_ij · (V_j · M_j)
    """
    def __init__(self, d_model, nhead, dropout=0.1, lambda_mask=3.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.lambda_mask = lambda_mask
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, mask_j):
        B, N, D = memory.shape
        Q = self.q_proj(query).view(B, 1, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(memory).view(B, N, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(memory).view(B, N, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        logits = Q @ K.transpose(-2, -1) * self.scale
        log_m = (mask_j + 1e-6).log().unsqueeze(1).unsqueeze(2)
        logits = logits + self.lambda_mask * log_m
        attn = logits.softmax(dim=-1)
        attn = self.dropout(attn)
        V_gated = V * mask_j.unsqueeze(1).unsqueeze(-1)
        out = (attn @ V_gated).permute(0, 2, 1, 3).reshape(B, 1, D)
        return self.out_proj(out)


class MaskBiasedDecoderLayer(nn.Module):
    """Decoder layer: self-attn → mask-biased cross-attn → FFN (norm_first)."""
    def __init__(self, d_model, nhead, ff_dim, dropout=0.1, lambda_mask=3.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MaskBiasedCrossAttn(d_model, nhead, dropout, lambda_mask)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, tgt, memory, mask_j):
        tgt = tgt + self.dropout(self.self_attn(self.n1(tgt), self.n1(tgt), self.n1(tgt), need_weights=False)[0])
        tgt = tgt + self.cross_attn(self.n2(tgt), memory, mask_j)
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
        self.grid_proj = nn.Linear(total_grid_dim, hidden_dim)

        # ── V10+ temporal mask mixer (across-frame at each grid location) ──
        if version in ("v10", "v11"):
            self.temporal_mask_mixer = TemporalMaskMixer(
                hidden_dim, nhead=max(4, n_heads // 2), num_layers=1,
                ff_dim=hidden_dim * 2, dropout=dropout,
            )
            self.pose_aux_head = nn.Linear(hidden_dim, 4)
        else:
            self.temporal_mask_mixer = None
            self.pose_aux_head = None

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
        self.joint_embed = nn.Parameter(torch.randn(self.num_joints, hidden_dim) * 0.02)
        self.mask_generator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

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
        nn.init.normal_(self.joint_embed, mean=0.0, std=0.02)
        # Shared mask generator: last bias = -1 → sigmoid(-1) ≈ 0.27
        for module in self.mask_generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.mask_generator[-1].bias, -1.0)
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
        # return_aux=True (V10 training only) additionally returns pose_aux [B,4].
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

        grid = torch.cat([grid_rgb, grid_elev], dim=-1)
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
        if self.temporal_mask_mixer is not None:
            grid = self.temporal_mask_mixer(grid)

        grid = self.pos_embed(grid)
        tokens = grid.reshape(B, T * G * G, D)
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # ── Joint-conditioned masks ──
        # mask_j = sigmoid(mask_heads[j](tokens + joint_embed[j]))
        # Each mask is explicitly tied to one joint through the embedding.
        masks_list = []
        for j in range(self.num_joints):
            cond = tokens + self.joint_embed[j]
            m_j = torch.sigmoid(self.mask_generator(cond)).squeeze(-1)
            masks_list.append(m_j)
        masks_flat = torch.stack(masks_list, dim=1)                 # [B, 4, N]
        masks_spatial = masks_flat.view(B, self.num_joints, T, G, G)

        # ── Soft union gate for shared encoder ──
        gate = 1.0 - (1.0 - masks_flat).prod(dim=1)                # [B, N]
        gate = 0.02 + 0.98 * gate
        gated_tokens = tokens * gate.unsqueeze(-1)

        # ── Bidirectional history encoder ──
        # Window of 8 past frames are all observed; no causal mask needed.
        memory = self.encoder(gated_tokens)

        # ── Mask-biased joint cross-attention decoder ──
        # Each joint query decodes through its own mask_j:
        #   attn_logits += λ·log(mask_j + ε), value *= mask_j
        decoded_list = []
        for j in range(self.num_joints):
            tgt = self.joint_queries[:, j:j+1, :].expand(B, -1, -1)
            for layer in self.decoder_layers:
                tgt = layer(tgt, memory, masks_flat[:, j, :])
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

        avg_masks = masks_spatial.mean(dim=2)                       # [B, 4, G, G]
        outputs = (action, avg_masks, masks_spatial)
        if return_aux and self.pose_aux_head is not None:
            # Pose auxiliary: predict current qpos from last-frame spatial tokens
            pose_aux = self.pose_aux_head(memory[:, -G * G:].mean(dim=1))
            return (*outputs, pose_aux)
        return outputs


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
