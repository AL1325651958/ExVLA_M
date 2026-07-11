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


class ExcavatorVLAYolo(nn.Module):
    def __init__(
        self, seq_len=8, img_size=224, hidden_dim=512,
        n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1,
        pretrained=True, num_excavators=4,
        use_sincos_output=True,
        qpos_mode="modulation",
        qpos_drop_prob=0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.use_sincos_output = use_sincos_output
        self.qpos_mode = qpos_mode
        self.qpos_drop_prob = qpos_drop_prob
        self.out_dim = 8 if use_sincos_output else 4
        out_dim = self.out_dim

        neck_out = 256
        grid_dim = 256

        self.neck = FPNPAN(p3_c=128, p4_c=256, p5_c=512, out_c=neck_out)
        self.rgb_backbone = CSPDarknet(3)
        self.rgb_head = SpatialGridHead(neck_out, grid_dim)
        self.elev_backbone = CSPDarknet(3)
        self.elev_head = SpatialGridHead(neck_out, grid_dim)

        grid_size = img_size // 16
        total_grid_dim = 3 * grid_dim * 2
        self.grid_size = grid_size
        self.grid_proj = nn.Linear(total_grid_dim, hidden_dim)

        self.num_regions = 4
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_regions),
        )

        self.num_excavators = num_excavators

        if qpos_mode == "modulation":
            self.qpos_mods = nn.ModuleList([
                nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, out_dim))
                for _ in range(num_excavators)
            ])
        elif qpos_mode == "transformer":
            self.qpos_proj = nn.Sequential(
                nn.Linear(4, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
        else:
            self.qpos_mods = None

        self.excv_embed = nn.Embedding(num_excavators, hidden_dim)
        self.pos_embed = SpatioTemporalPosEmbed(seq_len, grid_size, hidden_dim)

        self.num_queries = 4
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(2, n_layers // 2))

        # Per-excavator action heads — different machines have different joint ranges
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, out_dim),
            ) for _ in range(num_excavators)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        for head in self.action_heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.normal_(self.excv_embed.weight, mean=0.0, std=0.02)
        if self.qpos_mods is not None:
            for mod in self.qpos_mods:
                for module in mod:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def decode_action(self, raw):
        if self.use_sincos_output:
            sin, cos = raw.chunk(2, dim=-1)
            return torch.atan2(sin, cos)
        return raw

    def forward(self, rgb, elevation, qpos=None, excavator_id=None):
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]
        G = self.grid_size
        D = self.hidden_dim

        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)

        p3_r, p4_r, p5_r = self.rgb_backbone(rgb_flat)
        n3_r, n4_r, n5_r = self.neck(p3_r, p4_r, p5_r)
        grid_rgb = self.rgb_head(n3_r, n4_r, n5_r, G)

        p3_e, p4_e, p5_e = self.elev_backbone(elev_flat)
        n3_e, n4_e, n5_e = self.neck(p3_e, p4_e, p5_e)
        grid_elev = self.elev_head(n3_e, n4_e, n5_e, G)

        grid = torch.cat([grid_rgb, grid_elev], dim=-1)
        grid = self.grid_proj(grid).view(B, T, G, G, D)

        if self.qpos_mode == "transformer" and qpos is not None and self.training:
            if self.qpos_drop_prob > 0:
                m = (torch.rand(B, 1, 1, 1, 1, device=qpos.device) > self.qpos_drop_prob).float()
            else:
                m = 1.0
            grid = grid + self.qpos_proj(qpos).unsqueeze(2).unsqueeze(2) * m

        grid = self.pos_embed(grid)
        tokens = grid.reshape(B, T * G * G, D)
        if excavator_id is not None:
            tokens = tokens + self.excv_embed(excavator_id).unsqueeze(1)

        # ── Masks BEFORE encoder: causal gate on INPUT tokens ──
        # mask_head sees spatially-specific pre-encoder tokens and selects
        # task-relevant regions. Encoder is forced to route information ONLY
        # through selected tokens → masks must focus or prediction fails.
        raw_scores = self.mask_head(tokens)  # [B, T*G^2, 4]

        # Sigmoid activation: each region independently decides per-position relevance.
        # Unlike softmax, this allows positions to be "off" for ALL regions (background).
        region_acts = torch.sigmoid(raw_scores)  # [B, T*G^2, 4], each in [0,1]
        masks_flat = region_acts.permute(0, 2, 1)  # [B, 4, T*G^2]
        masks_spatial = masks_flat.view(B, self.num_regions, T, G, G)

        # Soft union gate via element-wise max: if ANY region activates a position,
        # that position passes through. Small floor prevents complete zero-out
        # which would cause LayerNorm NaN inside encoder.
        gate = masks_flat.max(dim=1).values  # [B, T*G^2], ∈ [0,1]
        gate = gate.clamp(min=0.02)
        gated_tokens = tokens * gate.unsqueeze(-1)

        # ── Causal mask for encoder self-attention ──
        # Token at time t can only attend to tokens at time ≤ t.
        # Within the same timestep: all G² spatial tokens see each other.
        # -1e9 is amp-safe (fp16 inf can cause NaN in softmax denominator).
        token_time = torch.arange(T * G * G, device=tokens.device) // (G * G)     # [N]
        causal_mask = (token_time.unsqueeze(1) < token_time.unsqueeze(0)).float() * (-1e9)

        memory = self.encoder(gated_tokens, mask=causal_mask)

        queries = self.query_tokens.expand(B, -1, -1)
        decoded = self.decoder(queries, memory)
        pool = decoded.mean(dim=1)                                          # [B, D]

        # Per-excavator action heads — each machine type has its own predictor
        action = torch.zeros(B, self.out_dim, device=pool.device, dtype=pool.dtype)
        for eid in range(self.num_excavators):
            mask_e = (excavator_id == eid)
            if mask_e.any():
                action[mask_e] = self.action_heads[eid](pool[mask_e])

        if self.qpos_mods is not None and qpos is not None and self.training:
            correction = torch.zeros(B, self.out_dim, device=action.device, dtype=action.dtype)
            for eid in range(self.num_excavators):
                mask_e = (excavator_id == eid)
                if mask_e.any():
                    correction[mask_e] = self.qpos_mods[eid](qpos[mask_e, -1, :])
            if self.qpos_drop_prob > 0:
                correction = correction * (torch.rand(B, 1, device=qpos.device) > self.qpos_drop_prob).float()
            action = action + correction

        avg_masks = masks_spatial.mean(dim=2)  # [B, 4, G, G] for viz
        return action, avg_masks, masks_spatial  # [B,4], [B,4,G,G], [B,4,T,G,G]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
