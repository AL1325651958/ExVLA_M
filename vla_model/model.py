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
    """Gated temporal-conv block with optional diagonal-SSM recurrence.

    Two modes, controlled by `d_state`:
      d_state=0 → pure Conv1d+Gate  (fast, stable, no overfitting risk)
      d_state>0 → Conv1d + SSM scan (selective state-space memory)

    The SSM is minimal by design to avoid overfitting:
      - d_state=4  (not 16)  — just enough for one "memory slot" per joint
      - dt is a single scalar per timestep shared across all channels
      - Binary associative scan for O(log T) gradient depth
      - A initialised for slow decay (τ ≈ 2-8 frames)
    """

    def __init__(self, d_model: int = 512, d_state: int = 0, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner = int(d_model * expand)

        # ── Gate + signal ──
        self.in_proj = nn.Linear(d_model, self.inner * 2, bias=False)

        # ── Causal depthwise conv ──
        self.conv1d = nn.Conv1d(
            in_channels=self.inner, out_channels=self.inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.inner,
        )

        # ── SSM (only if d_state > 0) ──
        if d_state > 0:
            # A: diagonal state transition,  A = -exp(A_log)
            #    Init: τ (time-constant in frames) ≈ 2, 4, 6, 8
            tau_frames = torch.linspace(2.0, 8.0, d_state, dtype=torch.float32)
            self.A_log = nn.Parameter((1.0 / tau_frames).log())
            # B, C: project inner → d_state  (per-channel selectivity)
            self.B_proj = nn.Linear(self.inner, d_state, bias=False)
            self.C_proj = nn.Linear(self.inner, d_state, bias=False)
            # dt: single scalar per timestep (shared across channels)
            self.dt_net = nn.Sequential(
                nn.Linear(self.inner, 1, bias=True),
            )
            # D skip
            self.D = nn.Parameter(torch.ones(self.inner))
            # Init dt bias so sigmoid ≈ 0.3 (moderate step)
            nn.init.constant_(self.dt_net[0].bias, -0.85)
            self.norm_ssm = nn.LayerNorm(self.inner)

        # ── Normalisation ──
        self.norm_in  = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        # ── Output projection ──
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    # ------------------------------------------------------------------
    #  Prefix scan via cumsum  (closed-form, O(1) gradient)
    # ------------------------------------------------------------------
    @staticmethod
    def _ssm_scan(dt: torch.Tensor, A: torch.Tensor,
                  B: torch.Tensor, C: torch.Tensor,
                  x: torch.Tensor) -> torch.Tensor:
        """Closed-form prefix scan for the recurrence h_t = A_bar·h_{t-1} + B_bar·x.

        Unrolling:  h[t] = Σ_{i≤t} exp(A·(cumΔ[t] - cumΔ[i])) · Δ[i]·B[i]·x[i]

        Args:
            dt:  [B, T, 1]     per-step scalar Δ  ∈ (0, 1)
            A:   [S]            per-state decay    < 0
            B:   [B, T, C, S]   input→state projection
            C:   [B, T, C, S]   state→output projection
            x:   [B, T, C]      signal after conv
        Returns:
            y:   [B, T, C]      SSM output
        """
        B, T, C, S = B.shape

        cum_dt = dt.cumsum(dim=1)                        # [B, T, 1]

        # term[t] = Δ[t] · B[t] · x[t]  →  [B, T, C, S]
        term = dt.unsqueeze(-1) * B * x.unsqueeze(-1)

        # decay[t] = exp(-A · cumΔ[t])   −A > 0
        decay = torch.exp(
            (-A).view(1, 1, 1, S) * cum_dt.unsqueeze(-1)
        )                                               # [B, T, 1, S]

        inner = torch.cumsum(decay * term, dim=1)        # [B, T, C, S]

        # h[t] = exp(A · cumΔ[t]) · inner[t]
        gain = torch.exp(
            A.view(1, 1, 1, S) * cum_dt.unsqueeze(-1)
        )                                               # [B, T, 1, S]
        h = gain * inner                                 # [B, T, C, S]

        return (C * h).sum(dim=-1)                       # [B, T, C]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x

        # 1 ── Gate + signal ──
        proj   = self.in_proj(self.norm_in(x))          # [B, T, inner*2]
        z, x_s = proj.chunk(2, dim=-1)                  # [B, T, inner]

        # 2 ── Causal conv ──
        x_c = self.conv1d(x_s.transpose(1, 2))[:, :, :T]
        x_c = F.silu(x_c.transpose(1, 2))               # [B, T, inner]

        # 3 ── SSM scan (only if d_state > 0) ──
        if self.d_state > 0:
            # scalar dt per timestep ∈ (0, 1)
            dt = torch.sigmoid(self.dt_net(x_c))        # [B, T, 1]
            # B, C: per-channel selectivity  [B, T, inner] → [B, T, ds]
            B_t = self.B_proj(x_c)                      # [B, T, ds]
            C_t = self.C_proj(x_c)                      # [B, T, ds]
            # A: diagonal decay  = -exp(A_log)  ∈ (-0.5, -0.125)
            A = -torch.exp(self.A_log)                  # [ds]

            # Broadcast B, C over inner dimension
            # cumsum scan: d_state stays as the contraction dim
            B_br = B_t.unsqueeze(2).expand(B, T, self.inner, self.d_state)
            C_br = C_t.unsqueeze(2).expand(B, T, self.inner, self.d_state)

            ssm_out = self._ssm_scan(dt, A, B_br, C_br, x_c)  # [B, T, inner]

            # skip + norm
            x_c = self.norm_ssm(ssm_out + x_c * self.D)

        # 4 ── Gate + output ──
        y = x_c * F.silu(z)                              # [B, T, inner]
        y = self.dropout(self.out_proj(y))               # [B, T, d_model]
        return self.norm_out(residual + y)


class MambaEncoder(nn.Module):
    """Stack of MambaBlocks — gated-conv (or gated-conv+SSM)."""

    def __init__(self, d_model: int = 512, n_layers: int = 4,
                 d_state: int = 0, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
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
#  Joint encoding helpers — encode radians → [sin, cos] for smooth circular rep
# ---------------------------------------------------------------------------

def encode_joints_sincos(qpos_rad: torch.Tensor) -> torch.Tensor:
    """Encode joint angles from radians to [sin(θ), cos(θ)] per joint.

    Args:
        qpos_rad: [...]  last dim=4 (joint angles in radians)
    Returns:
        [...]  last dim=8 ([sin₀,cos₀, sin₁,cos₁, sin₂,cos₂, sin₃,cos₃])
    """
    sin = torch.sin(qpos_rad)
    cos = torch.cos(qpos_rad)
    return torch.cat([sin, cos], dim=-1)


def decode_joints_sincos(sincos: torch.Tensor) -> torch.Tensor:
    """Decode from [sin, cos] back to radians via atan2.

    Args:
        sincos: [...]  last dim=8 (sin/cos pairs)
    Returns:
        [...]  last dim=4 (radians)
    """
    sin, cos = sincos.chunk(2, dim=-1)
    return torch.atan2(sin, cos)


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class ExcavatorVLA(nn.Module):
    """Vision-first VLA with swappable temporal encoder.

    Encoder options:
      - "transformer": PyTorch TransformerEncoder (quadratic attention)
      - "mamba":       Mamba-style linear-time SSM encoder

    Architecture:
        RGB + Elev → ResNet → [Transformer | Mamba] → vision_feat ─┐
                                                                    ├→ base_pred
        excv_id ───────────────────────────────────────────────────┘
        qpos[:, -1] → modulation MLP → correction (training only) ──┤
                                                                    ↓
                                                          final_pred [B, 4]
    """

    def __init__(
        self, seq_len=8, hidden_dim=512, n_heads=8, n_layers=4, ff_dim=2048,
        dropout=0.1, drop_path_rate=0.05, pretrained=True, num_excavators=4,
        qpos_mode: str = "modulation", qpos_drop_prob=0.3,
        encoder_type: str = "transformer",
        mamba_d_state: int = 0, mamba_d_conv: int = 4, mamba_expand: int = 2,
        use_sincos: bool = False,
    ):
        """
        Args:
            encoder_type: "transformer" or "mamba"
            qpos_mode:    "modulation" (head residual) or "transformer" (inject into encoder)
            use_sincos:   encode qpos as [sin(θ), cos(θ)] — eliminates 2π discontinuity
            mamba_d_state: 0 = Conv1d only, 4 = +minimal SSM (1 slot/joint)
            mamba_d_conv, mamba_expand: Mamba hyper-params
        """
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.qpos_mode = qpos_mode
        self.qpos_drop_prob = qpos_drop_prob
        self.encoder_type = encoder_type
        self.use_sincos = use_sincos

        # qpos dimension: 4 for raw radians, 8 for sin/cos pairs
        qpos_dim = 8 if use_sincos else 4

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
            # Mamba handles position natively — no explicit pos encoding needed
            self.encoder = MambaEncoder(
                d_model=hidden_dim, n_layers=n_layers,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ── Excavator-specific prediction heads ──
        # Shared vision → per-excavator MLP: each machine has its own kinematics.
        # vision_feat [D] → head[excv_id] → [4]
        head_in = hidden_dim  # vision_feat only (excv_id selects the head)
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, 4),
            )
            for _ in range(num_excavators)
        ])
        # Keep excv_embed for downstream compatibility (not used in forward)

        # ── qpos components ──
        if qpos_mode == "none":
            self.qpos_mod = None     # pure vision — no GT involvement
        elif qpos_mode == "modulation":
            self.qpos_mod = nn.Sequential(
                nn.Linear(qpos_dim, 32), nn.GELU(), nn.Linear(32, 4),
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
        if self.qpos_mode == "none":
            pass  # no qpos parameters to init
        elif self.qpos_mode == "modulation":
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
        """
        Returns:
            next_qpos [B, 4] — absolute joint angles (radians)
        """
        B, T = rgb.shape[:2]
        H, W = rgb.shape[3], rgb.shape[4]

        # ── 1. Vision encoder (per-frame) ──
        rgb_flat = rgb.reshape(B * T, 3, H, W)
        elev_flat = elevation.reshape(B * T, 3, H, W)
        features = self.vision_encoder(rgb_flat, elev_flat).view(B, T, -1)  # [B, T, D]

        # ── 2. qpos injection (transformer/mamba mode) — training only ──
        if self.qpos_mode == "transformer" and qpos is not None and self.training:
            qpos_enc = encode_joints_sincos(qpos) if self.use_sincos else qpos
            qpos_feat = self.qpos_proj(qpos_enc)
            if self.qpos_drop_prob > 0:
                mask = (torch.rand(B, 1, 1, device=qpos.device) > self.qpos_drop_prob).float()
                qpos_feat = qpos_feat * mask
            features = features + qpos_feat

        # ── 3. Temporal encoder ──
        if self.encoder_type == "transformer":
            # Position encoding
            if T <= self.seq_len:
                features = features + self.pos_embed[:, :T, :]
            else:
                features = self.sinusoidal_pe(features.permute(1, 0, 2)).permute(1, 0, 2)

            # Transformer: [T, B, D]
            features = features.permute(1, 0, 2)
            encoded = self.stoch_depth(self.encoder(features))
            encoded = encoded.permute(1, 0, 2)          # [B, T, D]
        else:
            # Mamba: [B, T, D] — no position encoding needed
            encoded = self.encoder(features)             # [B, T, D]

        # ── 4. Head: per-excavator MLP — each machine has its own kinematics ──
        vision_feat = encoded[:, -1, :]                  # [B, D]
        base_pred = torch.zeros(B, 4, device=vision_feat.device, dtype=vision_feat.dtype)
        for excv_idx in range(self.action_heads.__len__()):
            mask = (excavator_id == excv_idx)
            if mask.any():
                head_out = self.action_heads[excv_idx](vision_feat[mask].float())
                base_pred[mask] = head_out.to(vision_feat.dtype)

        # ── 5. qpos modulation residual (training only) ──
        if self.qpos_mode == "none":
            return base_pred
        elif self.qpos_mode == "modulation" and qpos is not None and self.training:
            qpos_last = encode_joints_sincos(qpos[:, -1, :]) if self.use_sincos else qpos[:, -1, :]
            correction = self.qpos_mod(qpos_last)
            if self.qpos_drop_prob > 0:
                mask = (torch.rand(B, 1, device=qpos.device) > self.qpos_drop_prob).float()
                correction = correction * mask
            return base_pred + correction
        else:
            return base_pred


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
