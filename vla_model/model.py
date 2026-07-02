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
    """Selective state-space block with bottlenecked diagonal SSM.

    Architecture (Mamba-inspired, bottlenecked for stability):
        x → LayerNorm → Conv1d → SiLU → proj_down(ssm_dim) ─┐
                                       ┌────────────────────┘
                                       ↓
                           SSM core (selective scan, d_state)
                                       ↓
                               norm → proj_up(inner) ─┐
                                                       ├→ ⊗SiLU(z) → out
        x → LayerNorm → in_proj → z (gate) ──────────┘

    The SSM operates in a low-dimension bottleneck (ssm_dim=128 by default)
    rather than the full inner=1024 space.  This keeps the serial-scan
    gradient chain narrow and numerically stable while still learning
    rich temporal selectivity.
    """

    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1,
                 ssm_bottleneck: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        inner = int(d_model * expand)
        self.inner = inner
        ssm_dim = ssm_bottleneck if ssm_bottleneck > 0 else inner

        # ── Input projection → gate + signal ──
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)

        # ── Causal depthwise conv (temporal mixing) ──
        self.conv1d = nn.Conv1d(
            in_channels=inner, out_channels=inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=inner,
        )

        # ── Bottleneck: inner → ssm_dim → inner ──
        self.ssm_down = nn.Linear(inner, ssm_dim, bias=False)
        self.ssm_up   = nn.Linear(ssm_dim, inner, bias=False)

        # ── Selective SSM core (operates on ssm_dim) ──
        # A: diagonal state-transition (log-space → -exp(·) ≤ 0, stable)
        A_init = torch.linspace(0.5, 3.0, d_state, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))
        # B(x), C(x): input-dependent projections
        self.B_proj = nn.Linear(ssm_dim, d_state, bias=False)
        self.C_proj = nn.Linear(ssm_dim, d_state, bias=False)
        # Δ(x): per-channel discretisation step
        self.dt_proj = nn.Linear(ssm_dim, ssm_dim, bias=True)
        # D: skip connection (1 per channel)
        self.D = nn.Parameter(torch.ones(ssm_dim))

        # ── Normalisation ──
        self.ssm_norm = nn.LayerNorm(ssm_dim)

        # ── Output ──
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Init dt bias: softplus ≈ 0.05 base rate  (small step, slow decay)
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            y: [B, T, d_model]
        """
        B, T, _ = x.shape
        inner = self.inner
        d_state = self.d_state

        residual = x

        # 1 ── Gate (z)  &  signal (x_sig) branches ──
        proj = self.in_proj(self.norm(x))            # [B, T, inner·2]
        z, x_sig = proj.chunk(2, dim=-1)             # [B, T, inner]

        # 2 ── Causal 1-D conv over time ──
        x_conv = self.conv1d(x_sig.transpose(1, 2))[:, :, :T]
        x_conv = F.silu(x_conv.transpose(1, 2))      # [B, T, inner]

        # 3 ── Bottleneck down: inner → ssm_dim ──
        x_ssm_in = self.ssm_down(x_conv)             # [B, T, ssm_dim]

        # 4 ── Discretisation step Δ(x) ∈ (0, ∞) ──
        dt = F.softplus(self.dt_proj(x_ssm_in))      # [B, T, ssm_dim]
        dt = dt.clamp(max=20.0)                      # prevent extreme values

        # 5 ── Selective B(x), C(x) ──
        B_sel = self.B_proj(x_ssm_in)                # [B, T, d_state]
        C_sel = self.C_proj(x_ssm_in)                # [B, T, d_state]

        # 6 ── Diagonal A = -exp(log) ≤ 0  ⇒  exp(Δt·A) ∈ (0,1] ──
        A_vec = -torch.exp(self.A_log)               # [d_state]

        # 7 ── Parallel prefix scan via cusum ──
        #      Unrolling the recurrence  h_t = Ā_t·h_{t-1} + B̄_t·x_t
        #      (where Ā=exp(Δ⊗A), B̄=Δ⊗B) yields the closed form:
        #
        #        h_t = Σ_{i≤t}  exp(A·(cumΔ[t] − cumΔ[i])) · Δ[i]·B[i]·x[i]
        #
        #      Computed in O(T) as two cumsums:
        #        1.  cumΔ = cumsum(Δt)
        #        2.  term = Δt · B_sel · x_ssm_in          [B,T,ssm,ds]
        #        3.  inner = cumsum(exp(−A·cumΔ) · term)
        #        4.  h[t]  = exp(A·cumΔ[t]) · inner[t]
        #        5.  y[t]  = C[t] · h[t]
        #
        #      Gradient flows through cumsum (O(1) backward),
        #      NOT through a serial for-loop (O(T) backward).
        # ─────────────────────────────────────────────────────
        cum_dt = torch.cumsum(dt, dim=1)             # [B, T, ssm_dim]

        # term[b,t,c,s] = dt[b,t,c] * B[b,t,s] * x_ssm_in[b,t,c]
        term = (dt.unsqueeze(-1)
                * B_sel.unsqueeze(2)
                * x_ssm_in.unsqueeze(-1))            # [B, T, ssm_dim, d_state]

        # exp(−A·cumΔ) — A_vec ≤ 0 so −A_vec ≥ 0
        decay = torch.exp(
            -A_vec.view(1, 1, 1, d_state) * cum_dt.unsqueeze(-1)
        )                                            # [B, T, ssm_dim, d_state]

        inner_sum = torch.cumsum(decay * term, dim=1)  # [B, T, ssm_dim, d_state]

        # h[t] = exp(A·cumΔ[t]) · inner_sum[t]
        h = (torch.exp(A_vec.view(1, 1, 1, d_state) * cum_dt.unsqueeze(-1))
             * inner_sum)                             # [B, T, ssm_dim, d_state]

        # y[t] = Σ_s C[t,s] · h[t,c,s]
        x_ssm = (C_sel.unsqueeze(2) * h).sum(dim=-1)  # [B, T, ssm_dim]

        # 8 ── Normalise + skip ──
        x_ssm = self.ssm_norm(x_ssm + x_ssm_in * self.D)

        # 9 ── Bottleneck up: ssm_dim → inner ──
        x_ssm_up = self.ssm_up(x_ssm)                # [B, T, inner]

        # 10 ── Gating  SiLU(z) ──
        y = x_ssm_up * F.silu(z)                     # [B, T, inner]

        # 11 ── Output projection + residual ──
        y = self.dropout(self.out_proj(y))           # [B, T, d_model]
        return self.norm(residual + y)


class MambaEncoder(nn.Module):
    """Stack of MambaBlocks — drop-in replacement for TransformerEncoder."""

    def __init__(self, d_model: int = 512, n_layers: int = 4, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1,
                 ssm_bottleneck: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, ssm_bottleneck)
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
        mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2,
        mamba_ssm_bottleneck: int = 128,
        use_sincos: bool = False,
    ):
        """
        Args:
            encoder_type: "transformer" or "mamba"
            qpos_mode:    "modulation" (head residual) or "transformer" (inject into encoder)
            use_sincos:   encode qpos as [sin(θ), cos(θ)] — eliminates 2π discontinuity
            mamba_d_state, mamba_d_conv, mamba_expand, mamba_ssm_bottleneck: Mamba hyper-params
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
                ssm_bottleneck=mamba_ssm_bottleneck,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ── Base prediction head ──
        head_in = hidden_dim + hidden_dim  # vision_feat + excv_feat
        self.action_head = nn.Sequential(
            nn.Linear(head_in, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4),
        )

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
        for module in self.action_head:
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

        # ── 4. Head: vision + excavator ID → base prediction ──
        vision_feat = encoded[:, -1, :]                  # [B, D]
        excv_feat = self.excv_embed(excavator_id)        # [B, D]
        base_pred = self.action_head(torch.cat([vision_feat, excv_feat], dim=-1))

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
