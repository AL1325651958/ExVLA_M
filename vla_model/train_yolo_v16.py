"""V16 training: dual-modality 8-mask + cross-modal attention.

V16 architecture features:
- Independent RGB and Elevation necks with cross-modal attention
- 8 per-joint masks (4 RGB + 4 Elevation) computed BEFORE fusion
- Swing joint: unmasked global context in decoder (rotation detection)
- No temporal mask mixer (replaced by cross-modal exchange)
- No pose auxiliary head (V10-only feature)

Checkpoints include "model_version": "v16".
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
from vla_model.model_yolo import ExcavatorVLAYolo, count_parameters
from vla_model.dataset import ExcavatorDataset


# ── Shared utilities ──

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
    # Swing: circular variance
    s = np.asarray(swing_labels)
    sin_m, cos_m = np.sin(s).mean(), np.cos(s).mean()
    var_circ = 1.0 - np.sqrt(sin_m ** 2 + cos_m ** 2)
    var_circ = np.maximum(var_circ, 1e-10)
    r2[3] = 1 - ss_res[3] / (var_circ * n)
    return r2, float(r2.mean())


def _rad_to_output(rad: torch.Tensor, out_dims=(2, 2, 2, 2)) -> torch.Tensor:
    sin = torch.sin(rad)
    cos = torch.cos(rad)
    return torch.stack([sin, cos], dim=-1).reshape(rad.size(0), -1)


def build_v16_model(seq_len=8, img_size=224, hidden_dim=512, n_heads=8,
                     n_layers=4, ff_dim=2048, dropout=0.1, pretrained=True,
                     num_excavators=4):
    """Construct a V16 model: cross-modal attention + 8 independent masks."""
    return ExcavatorVLAYolo(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
        dropout=dropout, pretrained=pretrained, num_excavators=num_excavators,
        version="v16",
    )


# ── V16 training ──

def train_epoch(model, dataloader, optimizer, scaler, scheduler, criterion, config, epoch):
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

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)

        optimizer.zero_grad()

        with torch.amp.autocast(config.device):
            # V16: 3-output path (no pose_aux)
            raw_out, masks_avg, masks_spatial = model(
                rgb, elevation, qpos, excavator_id
            )
            action_gt_rad = action_gt.squeeze(1)
            target = _rad_to_output(action_gt_rad, out_dims)

            # 1. Prediction loss
            pred_loss = criterion(raw_out, target)

            # 2. Unit-circle constraint
            raw_4d = raw_out.view(raw_out.size(0), 4, 2)
            circle_err = (raw_4d.pow(2).sum(dim=-1) - 1.0).pow(2).mean()
            circle_loss = 0.1 * circle_err

            # 3. Per-joint target area
            # V16: masks_spatial = [B, 2, 4, T, G, G] (modality × joint × time × spatial)
            # Compute per-joint union mask (RGB ∪ Elev), then area per joint per sample.
            # Boom/Arm/Bucket: [0.05, 0.30]  —  local features
            # Swing:           [0.10, 0.70]  —  rotation needs global context
            union_masks = 1.0 - (1.0 - masks_spatial[:, 0]) * (1.0 - masks_spatial[:, 1])  # [B, 4, T, G, G]
            area_per_joint = union_masks.mean(dim=(-2, -1))                                  # [B, 4, T] → mean over T
            area_per_joint = area_per_joint.mean(dim=-1)                                       # [B, 4]
            # Boom/Arm/Bucket (indices 0,1,2)
            area_planar = area_per_joint[:, :3]
            area_loss_planar = (torch.relu(0.05 - area_planar).pow(2) +
                                torch.relu(area_planar - 0.30).pow(2)).mean()
            # Swing (index 3) — wider range, lower weight
            area_swing = area_per_joint[:, 3]
            area_loss_swing = (torch.relu(0.10 - area_swing).pow(2) +
                               torch.relu(area_swing - 0.70).pow(2)).mean()
            area_loss = 1.0 * area_loss_planar + 0.2 * area_loss_swing

            # 4. Cosine-similarity overlap with margin
            # V16: [B, 2, 4, T, G, G] → flatten to [B, 8, T*G*G] for all-mask diversity
            # Use .mean() not .sum() — scale-invariant to number of masks.
            B_s = masks_spatial.size(0)
            mf = masks_spatial.reshape(B_s, 8, -1)
            norms = mf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mf_n = mf / norms
            cos_sim = torch.bmm(mf_n, mf_n.transpose(1, 2))
            eye = torch.eye(8, device=cos_sim.device).unsqueeze(0)
            off_diag = cos_sim * (1 - eye)
            margin = 0.3
            diversity_loss = 0.5 * torch.relu(off_diag - margin).pow(2).mean()

            # 5. Temporal smoothness — slice TIME dim (dim=3), NOT joint dim (dim=2)
            temp_diff = (masks_spatial[:, :, :, 1:] - masks_spatial[:, :, :, :-1]).abs().mean()
            temp_loss = 0.005 * temp_diff

            loss = pred_loss + circle_loss + area_loss + diversity_loss + temp_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
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
    """V16 validation: 3-output path."""
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
        loss = criterion(raw_out, target)

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


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics, config, is_best=False):
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "best" if is_best else f"epoch_{epoch+1}"
    path = os.path.join(config.output_dir, f"yolo_v16_checkpoint_{suffix}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "model_version": "v16",
    }, path)
    print(f"Checkpoint saved: {path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Train ExcavatorVLA-YOLO V16")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--exclude_306", action="store_true",
                        help="Exclude excavator 306 (nighttime) from training")
    args = parser.parse_args()

    config = Config()

    for key in ("data_dir", "epochs", "batch_size", "lr", "seq_len",
                "sample_ratio", "img_size", "output_dir"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}")
    print(f"V16: cross-modal attention + 8 independent masks (RGB×4 + Elev×4)")
    print(f"     Swing decoder: unmasked global context")
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

    # ── V16 Model ──
    model = build_v16_model(
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location=config.device, weights_only=False)
        resume_state = ckpt["model_state_dict"]
        model_state = model.state_dict()
        filtered_state = {}
        skipped = 0
        for k, v in resume_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped += 1
        model.load_state_dict(filtered_state, strict=False)
        print(f"  [resume] loaded {len(filtered_state)} keys, skipped {skipped} (size mismatch or missing)")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"  [resume] scheduler LR restored to {scheduler.get_last_lr()[0]:.2e}")
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [],
               "train_r2": [], "val_r2": []}

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, scheduler,
                                     criterion, config, epoch)

        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(config.ema_decay).add_(p.data, alpha=1 - config.ema_decay)

        val_metrics = {"loss": float("nan"), "mae": [0, 0, 0, 0], "mae_mean": float("nan"),
                       "r2": [0, 0, 0, 0], "r2_mean": float("nan")}
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae_mean"])
        history["train_r2"].append(train_metrics["r2_mean"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae_mean"])
        history["val_r2"].append(val_metrics["r2_mean"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Train Loss: {train_metrics['loss']:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Train MAE: {train_metrics['mae_mean']:.4f} | "
              f"Val MAE: {val_metrics['mae_mean']:.4f} | "
              f"Train R²: {train_metrics['r2_mean']:.4f} | "
              f"Val R²: {val_metrics['r2_mean']:.4f} | "
              f"Time: {elapsed:.0f}s")
        print(f"  Per-joint MAE - "
              f"Train: {[f'{x:.4f}' for x in train_metrics['mae']]} | "
              f"Val:   {[f'{x:.4f}' for x in val_metrics['mae']]}")
        print(f"  Per-joint R²  - "
              f"Train: {[f'{x:.4f}' for x in train_metrics['r2']]} | "
              f"Val:   {[f'{x:.4f}' for x in val_metrics['r2']]}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, config, is_best=True)

        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_metrics, config)

    save_checkpoint(model, optimizer, scaler, scheduler, config.epochs - 1, val_metrics, config)
    print(f"\nTraining complete.  Best val loss: {best_loss:.6f}")

    with open(os.path.join(config.output_dir, "yolo_v16_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
