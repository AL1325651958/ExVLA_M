"""V17.1 training: V17 independent masks + restored V10 temporal/pose supervision.

V17.1 = V17's 4 independent mask heads + graded per-joint decoder
     + V10's TemporalMaskMixer (restored)
     + V10's pose_aux qpos supervision (restored, training-only)
     + Swing velocity auxiliary loss (wrap-aware)
     + Swing-best checkpoint selection (saved alongside overall best)

Checkpoints include "model_version": "v17.1".
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.config import Config
from vla_model.model_yolo import (
    ExcavatorVLAYolo,
    count_parameters,
    load_compatible_state_dict,
    upgrade_legacy_v17_1_state_dict as _upgrade_legacy_v17_1_state_dict,
)
from vla_model.dataset import ExcavatorDataset


# ── Shared utilities ──

_TRAINING_VARIANTS = {
    "v17.1": {
        "model_version": "v17.1",
        "checkpoint_prefix": "yolo_v17_1",
        "diversity_mode": "legacy_all_pairs",
        "diversity_margin": 0.3,
        "mask_diagnostics": False,
    },
    "v17.3": {
        "model_version": "v17.3",
        "checkpoint_prefix": "yolo_v17_3",
        "diversity_mode": "within_modality",
        "diversity_margin": 0.5,
        "mask_diagnostics": True,
    },
}


def get_training_variant(version):
    """Return an isolated copy of one supported training-variant config."""
    try:
        return dict(_TRAINING_VARIANTS[str(version).lower()])
    except KeyError as error:
        supported = ", ".join(sorted(_TRAINING_VARIANTS))
        raise ValueError(
            f"unknown training variant {version!r}; expected one of: {supported}"
        ) from error


def _validate_dual_mask_shape(masks_spatial):
    if (
        masks_spatial.ndim != 6
        or masks_spatial.shape[1] != 2
        or masks_spatial.shape[2] != 4
    ):
        raise ValueError(
            "masks_spatial must have shape [B, 2, 4, T, G, G]"
        )


def compute_mask_diversity_loss(masks_spatial, mode, margin):
    """Penalize mask collapse using legacy or V17.3 pair selection."""
    _validate_dual_mask_shape(masks_spatial)
    if mode == "legacy_all_pairs":
        batch = masks_spatial.size(0)
        flat = masks_spatial.reshape(batch, 8, -1)
        normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        eye = torch.eye(
            8, device=similarity.device, dtype=similarity.dtype
        ).unsqueeze(0)
        off_diagonal = similarity * (1.0 - eye)
        return 0.5 * torch.relu(off_diagonal - margin).pow(2).mean()
    if mode == "within_modality":
        flat = masks_spatial.flatten(start_dim=3)
        normalized = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal = ~torch.eye(
            4, device=similarity.device, dtype=torch.bool
        )
        penalties = torch.relu(similarity - margin).pow(2)
        return 0.5 * penalties[..., off_diagonal].mean()
    raise ValueError(f"unknown mask diversity mode: {mode!r}")

def _circular_error(pred_rad, gt_rad):
    """Wrap-aware element-wise difference in [-π, π] (torch or numpy)."""
    delta = pred_rad - gt_rad
    if isinstance(delta, torch.Tensor):
        return torch.atan2(torch.sin(delta), torch.cos(delta))
    return np.arctan2(np.sin(delta), np.cos(delta))


def _compute_r2_circular(ss_res, sum_y, sum_y2, n, swing_labels):
    """R² per joint [4]. Swing (index 3) uses circular (wrap-aware) statistics."""
    r2 = np.zeros(4)
    for j in range(3):
        ss_tot = sum_y2[j] - (sum_y[j] ** 2) / n
        ss_tot = np.maximum(ss_tot, 1e-10)
        r2[j] = 1 - ss_res[j] / ss_tot
    # Swing: use wrapped squared deviations in both numerator and denominator.
    # The previous denominator used 1-|mean(exp(i*y))|, which is roughly half
    # the angular variance for concentrated labels and is not commensurate
    # with the squared-radian residual accumulated in ``ss_res``.
    s = np.asarray(swing_labels, dtype=np.float64)
    mean_angle = np.arctan2(np.sin(s).mean(), np.cos(s).mean())
    centered = _circular_error(s, mean_angle)
    ss_tot_swing = np.maximum(np.square(centered).sum(), 1e-10)
    r2[3] = 1 - ss_res[3] / ss_tot_swing
    return r2, float(r2.mean())


def _rad_to_output(rad: torch.Tensor, out_dims=(2, 2, 2, 2)) -> torch.Tensor:
    sin = torch.sin(rad)
    cos = torch.cos(rad)
    return torch.stack([sin, cos], dim=-1).reshape(rad.size(0), -1)


def _weighted_sincos_loss(prediction: torch.Tensor, target: torch.Tensor,
                          swing_weight: float = 2.0) -> torch.Tensor:
    """MSE over four sin/cos pairs with explicit Swing emphasis."""
    if prediction.shape != target.shape or prediction.shape[-1] != 8:
        raise ValueError("prediction and target must both have shape [B, 8]")
    per_joint = (prediction.view(-1, 4, 2) - target.view(-1, 4, 2)).pow(2)
    per_joint = per_joint.mean(dim=(0, 2))
    weights = prediction.new_tensor([1.0, 1.0, 1.0, float(swing_weight)])
    return (per_joint * weights).sum() / weights.sum()


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update EMA parameters per optimizer step and copy running buffers."""
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer.detach())


def restore_training_state(optimizer, scaler, scheduler, checkpoint):
    """Restore optimizer/scheduler atomically enough to avoid a stale LR."""
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        print("  [resume] optimizer state missing; using fresh optimizer and scheduler")
        return False
    try:
        optimizer.load_state_dict(optimizer_state)
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    except (ValueError, KeyError) as error:
        print(f"  [resume] optimizer/scaler state incompatible ({error}), "
              "using fresh optimizer and scheduler")
        return False

    if "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  [resume] scheduler LR restored to {scheduler.get_last_lr()[0]:.2e}")
        except (ValueError, KeyError):
            print("  [resume] scheduler state incompatible, using fresh scheduler")
    return True


def prepare_resume(checkpoint, weights_only=False):
    """Select checkpoint weights and metadata for full or weights-only resume."""
    ema_state = checkpoint["model_state_dict"]
    selected_state = (
        ema_state
        if weights_only
        else checkpoint.get("raw_model_state_dict", ema_state)
    )
    had_legacy_masks = (
        "mask_linear1.weight" in selected_state
        and "mask_linear1.0.weight" not in selected_state
    )
    ema_state = _upgrade_legacy_v17_1_state_dict(ema_state)
    selected_state = _upgrade_legacy_v17_1_state_dict(selected_state)

    if weights_only:
        start_epoch = 0
        best_loss = float("inf")
        best_swing_r2 = -float("inf")
    else:
        start_epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        best_loss = checkpoint.get("best_loss", metrics.get("loss", float("inf")))
        previous_r2 = metrics.get("r2", [0.0, 0.0, 0.0, -float("inf")])
        best_swing_r2 = checkpoint.get("best_swing_r2", previous_r2[3])

    return {
        "model_state_dict": selected_state,
        "ema_state_dict": ema_state,
        "start_epoch": start_epoch,
        "best_loss": best_loss,
        "best_swing_r2": best_swing_r2,
        "restore_training_state": not weights_only,
        "had_legacy_masks": had_legacy_masks,
    }


def build_v17_1_model(seq_len=8, img_size=224, hidden_dim=512, n_heads=8,
                       n_layers=3, ff_dim=2048, dropout=0.25, pretrained=True,
                       num_excavators=4):
    """Construct a V17.1 model: V17 masks + V10 temporal/poster supervision."""
    return ExcavatorVLAYolo(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
        dropout=dropout, pretrained=pretrained, num_excavators=num_excavators,
        version="v17.1",
    )


# ── V17.1 training ──

def train_epoch(model, dataloader, optimizer, scaler, scheduler, criterion, config, epoch,
                ema_model=None):
    model.train()
    total_loss = 0.0
    total_mae = np.zeros(4)
    ss_res = np.zeros(4)
    sum_y  = np.zeros(4)
    sum_y2 = np.zeros(4)
    n_total = 0
    n_batches = 0
    swing_labels = []  # collect Swing labels for circular R²
    out_dims = getattr(model, 'out_dims', (2, 2, 2, 2))

    pose_aux_weight = getattr(config, 'v17_pose_aux_weight', 0.05)
    vel_aux_weight   = getattr(config, 'v17_vel_aux_weight', 0.10)
    swing_loss_weight = getattr(config, 'v17_swing_loss_weight', 2.0)

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)

        optimizer.zero_grad()

        with torch.amp.autocast(config.device):
            # V17.1: return_aux=True for pose_aux + temporal_mask_mixer
            raw_out, masks_avg, masks_spatial, pose_aux, swing_vel_aux = model(
                rgb, elevation, qpos, excavator_id, return_aux=True
            )
            action_gt_rad = action_gt.squeeze(1)
            target = _rad_to_output(action_gt_rad, out_dims)

            # 1. Prediction loss
            pred_loss = _weighted_sincos_loss(
                raw_out, target, swing_weight=swing_loss_weight
            )

            # 2. Unit-circle constraint
            raw_4d = raw_out.view(raw_out.size(0), 4, 2)
            circle_err = (raw_4d.pow(2).sum(dim=-1) - 1.0).pow(2).mean()
            circle_loss = 0.1 * circle_err

            # 3. Per-joint target area
            union_masks = 1.0 - (1.0 - masks_spatial[:, 0]) * (1.0 - masks_spatial[:, 1])  # [B, 4, T, G, G]
            area_per_joint = union_masks.mean(dim=(-2, -1))                                  # [B, 4, T]
            area_per_joint = area_per_joint.mean(dim=-1)                                       # [B, 4]
            area_planar = area_per_joint[:, :3]
            area_loss_planar = (torch.relu(0.05 - area_planar).pow(2) +
                                torch.relu(area_planar - 0.30).pow(2)).mean()
            area_swing = area_per_joint[:, 3]
            area_loss_swing = (torch.relu(0.10 - area_swing).pow(2) +
                               torch.relu(area_swing - 0.70).pow(2)).mean()
            area_loss = 1.0 * area_loss_planar + 0.2 * area_loss_swing

            # 4. Cosine-similarity overlap with variant-specific pair selection
            diversity_loss = compute_mask_diversity_loss(
                masks_spatial,
                mode=getattr(
                    config, "v17_mask_diversity_mode", "legacy_all_pairs"
                ),
                margin=getattr(config, "v17_mask_diversity_margin", 0.3),
            )

            # 5. Temporal smoothness — slice TIME dim (dim=3), NOT joint dim (dim=2)
            temp_diff = (masks_spatial[:, :, :, 1:] - masks_spatial[:, :, :, :-1]).abs().mean()
            temp_loss = 0.005 * temp_diff

            # 6. Periodic pose auxiliary: current four-joint pose as sin/cos.
            # qpos is a training label only and is never consumed by forward.
            pose_aux_target = _rad_to_output(qpos[:, -1], out_dims)
            pose_aux_loss = criterion(pose_aux, pose_aux_target)

            # 7. Swing velocity auxiliary: wrap-aware angular velocity
            # GT Swing velocity between last input qpos and next action frame
            swing_vel_gt = _circular_error(action_gt_rad[:, 3], qpos[:, -1, 3])
            # Independent velocity head predicts a periodic sin/cos pair.
            swing_vel_target = torch.stack(
                [torch.sin(swing_vel_gt), torch.cos(swing_vel_gt)], dim=-1
            )
            vel_aux_loss = criterion(swing_vel_aux, swing_vel_target)
            vel_circle_error = (
                swing_vel_aux.pow(2).sum(dim=-1) - 1.0
            ).pow(2).mean()
            vel_aux_loss = vel_aux_loss + 0.1 * vel_circle_error

            # ── Mask regularization decay ──
            reg_scale = 1.0 if epoch < 40 else max(0.1, 1.0 - 0.9 * (epoch - 39) / max(1, config.epochs - 40))

            loss = (pred_loss + circle_loss
                    + reg_scale * (area_loss + diversity_loss + temp_loss)
                    + pose_aux_weight * pose_aux_loss
                    + vel_aux_weight * vel_aux_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema_model is not None:
            update_ema(ema_model, model, config.ema_decay)
        scheduler.step()       # per-batch LR scheduling

        total_loss += loss.item()
        action_pred_rad = model.decode_action(raw_out.detach())
        # Per-joint MAE: Boom/Arm/Bucket use linear error; Swing uses circular (wrap-aware)
        mae_linear = (action_pred_rad - action_gt_rad).abs()
        mae_linear[..., 3] = _circular_error(action_pred_rad[..., 3], action_gt_rad[..., 3]).abs()
        mae = mae_linear.mean(dim=0).cpu().numpy()
        total_mae += mae

        pred_cpu = action_pred_rad.cpu().numpy()
        label_cpu = action_gt_rad.cpu().numpy()
        # Squared residuals: Swing uses circular (wrap-aware) error
        ss_res_batch = (pred_cpu - label_cpu) ** 2
        ss_res_batch[:, 3] = _circular_error(pred_cpu[:, 3], label_cpu[:, 3]) ** 2
        ss_res += ss_res_batch.sum(axis=0)
        sum_y  += label_cpu.sum(axis=0)
        sum_y2 += (label_cpu ** 2).sum(axis=0)
        n_total += len(label_cpu)
        n_batches += 1
        swing_labels.extend(label_cpu[:, 3].tolist())

        if step % config.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "mae": f"{mae.mean():.4f}"})

    r2_per, r2_mean = _compute_r2_circular(ss_res, sum_y, sum_y2, n_total, swing_labels)
    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
        "r2": r2_per.tolist(),
        "r2_mean": r2_mean,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, config):
    """V17.1 validation: 3-output path (no aux)."""
    model.eval()
    total_loss = 0.0
    total_mae = np.zeros(4)
    ss_res = np.zeros(4)
    sum_y  = np.zeros(4)
    sum_y2 = np.zeros(4)
    n_total = 0
    n_batches = 0
    swing_labels = []  # collect Swing labels for circular R²
    out_dims = getattr(model, 'out_dims', (2, 2, 2, 2))

    for batch in tqdm(dataloader, desc="Validating"):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)

        raw_out, _, _ = model(rgb, elevation, qpos, excavator_id)
        action_gt_rad = action_gt.squeeze(1)
        target = _rad_to_output(action_gt_rad, out_dims)
        loss = _weighted_sincos_loss(
            raw_out, target,
            swing_weight=getattr(config, 'v17_swing_loss_weight', 2.0),
        )

        total_loss += loss.item()
        action_pred_rad = model.decode_action(raw_out)
        # Per-joint MAE: Swing uses circular (wrap-aware) error
        mae_linear = (action_pred_rad - action_gt_rad).abs()
        mae_linear[..., 3] = _circular_error(action_pred_rad[..., 3], action_gt_rad[..., 3]).abs()
        mae = mae_linear.mean(dim=0).cpu().numpy()
        total_mae += mae

        pred_cpu = action_pred_rad.cpu().numpy()
        label_cpu = action_gt_rad.cpu().numpy()
        ss_res_batch = (pred_cpu - label_cpu) ** 2
        ss_res_batch[:, 3] = _circular_error(pred_cpu[:, 3], label_cpu[:, 3]) ** 2
        ss_res += ss_res_batch.sum(axis=0)
        sum_y  += label_cpu.sum(axis=0)
        sum_y2 += (label_cpu ** 2).sum(axis=0)
        n_total += len(label_cpu)
        n_batches += 1
        swing_labels.extend(label_cpu[:, 3].tolist())

    r2_per, r2_mean = _compute_r2_circular(ss_res, sum_y, sum_y2, n_total, swing_labels)
    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
        "r2": r2_per.tolist(),
        "r2_mean": r2_mean,
    }


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics, config,
                    is_best=False, suffix_override=None, ema_model=None,
                    best_loss=float("inf"), best_swing_r2=-float("inf")):
    os.makedirs(config.output_dir, exist_ok=True)
    if suffix_override:
        suffix = suffix_override
    elif is_best:
        suffix = "best"
    else:
        suffix = f"epoch_{epoch+1}"
    path = os.path.join(config.output_dir, f"yolo_v17_1_checkpoint_{suffix}.pt")
    checkpoint_model = ema_model if ema_model is not None else model
    payload = {
        "epoch": epoch + 1,
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "best_loss": best_loss,
        "best_swing_r2": best_swing_r2,
        "config": config,
        "model_version": "v17.1",
    }
    if ema_model is not None:
        payload["raw_model_state_dict"] = model.state_dict()
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")
    return path


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Train ExcavatorVLA-YOLO V17.1")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights_only", action="store_true",
                        help="Load checkpoint EMA model weights but reset epoch, optimizer, and scheduler")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--exclude_306", action="store_true",
                        help="Exclude excavator 306 (nighttime) from training")
    parser.add_argument("--pose_aux_weight", type=float, default=0.05,
                        help="Weight for training-only qpos auxiliary loss")
    parser.add_argument("--vel_aux_weight", type=float, default=0.10,
                        help="Weight for Swing velocity auxiliary loss")
    parser.add_argument("--swing_loss_weight", type=float, default=2.0,
                        help="Relative weight of Swing in the main sin/cos loss")
    args = parser.parse_args()
    if args.weights_only and not args.resume:
        parser.error("--weights_only requires --resume CHECKPOINT")

    config = Config()

    for key in ("data_dir", "epochs", "batch_size", "lr", "seq_len",
                "sample_ratio", "img_size", "output_dir"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}")
    print(f"V17.1: V17 independent masks + V10 TemporalMaskMixer + pose_aux + Swing vel_aux")
    print(f"       Boom(λ_m=1.5/λ_v=0.8) Arm(λ_m=2.0/λ_v=1.0) Bucket(λ_m=2.5/λ_v=1.0) Swing(λ_m=0/λ_v=0.5)")
    print(f"       Anti-overfit: n_layers=3  dropout=0.25  wd=1e-3  mask_reg_decay(40→0.1x)")
    print(f"       pose_aux_weight={args.pose_aux_weight}  vel_aux_weight={args.vel_aux_weight}  "
          f"swing_loss_weight={args.swing_loss_weight}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}, indent=2)}")

    # ── Datasets ──
    train_dataset = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="train", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
        exclude_excv={1} if getattr(args, "exclude_306", False) else None,
    )
    val_dataset = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="val", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
        exclude_excv={1} if getattr(args, "exclude_306", False) else None,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True) if len(val_dataset) > 0 else None

    # ── V17.1 Model ──
    config.n_layers = 3
    config.dropout   = 0.25
    model = build_v17_1_model(
        seq_len=config.seq_len, img_size=config.img_size,
        hidden_dim=config.hidden_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ff_dim=config.ff_dim,
        dropout=config.dropout, pretrained=config.pretrained,
    ).to(config.device)

    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    G = config.img_size // 16
    print(f"  Grid size: {G}×{G}")
    print(f"  Tokens per sequence: {config.seq_len} × {G}² grid + {model.num_joints} queries = "
          f"{config.seq_len * G ** 2 + model.num_joints}")

    # ── Optimiser + scheduler ──
    config.weight_decay = 1e-3
    config.v17_pose_aux_weight = args.pose_aux_weight
    config.v17_vel_aux_weight  = args.vel_aux_weight
    config.v17_swing_loss_weight = args.swing_loss_weight
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler   = torch.amp.GradScaler(config.device)
    criterion = nn.MSELoss()

    # ── EMA ──
    ema_model = None
    if config.use_ema:
        ema_model = deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # ── Resume ──
    start_epoch = 0
    best_loss = float("inf")
    best_swing_r2 = -float("inf")
    resume_ema_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=config.device, weights_only=False)
        prepared = prepare_resume(ckpt, weights_only=args.weights_only)
        resume_ema_state = prepared["ema_state_dict"]
        if prepared["had_legacy_masks"]:
            print("  [resume] migrated shared V17.1 masks to four independent heads")
        loaded, skipped = load_compatible_state_dict(model, prepared["model_state_dict"])
        print(f"  [resume] loaded {loaded} keys, skipped {skipped} (size mismatch or missing)")
        if prepared["restore_training_state"]:
            restore_training_state(optimizer, scaler, scheduler, ckpt)
        start_epoch = prepared["start_epoch"]
        best_loss = prepared["best_loss"]
        best_swing_r2 = prepared["best_swing_r2"]
        if args.weights_only:
            print(f"Weights-only warm start from {args.resume}: epoch=0, fresh optimizer/scheduler")
        else:
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # EMA must start from the actually loaded live weights, not the freshly
    # constructed pre-resume model.
    if ema_model is not None:
        ema_model.load_state_dict(model.state_dict())
        if resume_ema_state is not None:
            loaded, skipped = load_compatible_state_dict(ema_model, resume_ema_state)
            print(f"  [resume] restored EMA: loaded {loaded} keys, skipped {skipped}")

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [],
               "train_r2": [], "val_r2": [], "val_swing_r2": []}

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, scheduler,
                                     criterion, config, epoch, ema_model=ema_model)

        val_metrics = {"loss": float("nan"), "mae": [0, 0, 0, 0], "mae_mean": float("nan"),
                       "r2": [0, 0, 0, 0], "r2_mean": float("nan")}
        if val_loader is not None and len(val_loader) > 0:
            eval_model = ema_model if ema_model is not None else model
            val_metrics = validate(eval_model, val_loader, criterion, config)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae_mean"])
        history["train_r2"].append(train_metrics["r2_mean"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae_mean"])
        history["val_r2"].append(val_metrics["r2_mean"])
        history["val_swing_r2"].append(val_metrics["r2"][3])

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Train Loss: {train_metrics['loss']:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Train MAE: {train_metrics['mae_mean']:.4f} | "
              f"Val MAE: {val_metrics['mae_mean']:.4f} | "
              f"Train R²: {train_metrics['r2_mean']:.4f} | "
              f"Val R²: {val_metrics['r2_mean']:.4f} | "
              f"Val swing R²: {val_metrics['r2'][3]:.4f} | "
              f"Time: {elapsed:.0f}s")
        print(f"  Per-joint MAE - "
              f"Train: {[f'{x:.4f}' for x in train_metrics['mae']]} | "
              f"Val:   {[f'{x:.4f}' for x in val_metrics['mae']]}")
        print(f"  Per-joint R²  - "
              f"Train: {[f'{x:.4f}' for x in train_metrics['r2']]} | "
              f"Val:   {[f'{x:.4f}' for x in val_metrics['r2']]}")

        improved_loss = val_metrics["loss"] < best_loss
        improved_swing = val_metrics["r2"][3] > best_swing_r2
        if improved_loss:
            best_loss = val_metrics["loss"]
        if improved_swing:
            best_swing_r2 = val_metrics["r2"][3]

        if improved_loss:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, config,
                            is_best=True, ema_model=ema_model, best_loss=best_loss,
                            best_swing_r2=best_swing_r2)

        if improved_swing:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, config,
                            is_best=False, suffix_override="best_swing", ema_model=ema_model,
                            best_loss=best_loss, best_swing_r2=best_swing_r2)

        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, config,
                            ema_model=ema_model, best_loss=best_loss,
                            best_swing_r2=best_swing_r2)

    save_checkpoint(model, optimizer, scaler, scheduler, config.epochs - 1, val_metrics, config,
                    ema_model=ema_model, best_loss=best_loss,
                    best_swing_r2=best_swing_r2)
    print(f"\nTraining complete.  Best val loss: {best_loss:.6f}  Best Swing R²: {best_swing_r2:.4f}")

    with open(os.path.join(config.output_dir, "yolo_v17_1_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
