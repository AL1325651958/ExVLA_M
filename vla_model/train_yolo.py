"""Training script for ExcavatorVLA-YOLO (spatio-temporal grid model).

Works with the same dataset and config. Just imports model_yolo instead of model.
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


# ---------------------------------------------------------------------------
#  Training utilities  (same as train.py)
# ---------------------------------------------------------------------------

def _delta_to_sincos(delta_rad: torch.Tensor) -> torch.Tensor:
    """[B, 4] rad → [B, 8] sin/cos pairs."""
    return torch.cat([torch.sin(delta_rad), torch.cos(delta_rad)], dim=-1)


def train_epoch(model, dataloader, optimizer, scaler, criterion, config, epoch):
    model.train()
    total_loss = 0.0
    total_mae = np.zeros(4)
    n_batches = 0
    use_sincos = getattr(model, 'use_sincos_output', False)
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)  # [B, 1, 4] raw radians delta

        optimizer.zero_grad()

        with autocast():
            raw_out, masks = model(rgb, elevation, qpos, excavator_id)  # [B, out_dim], [B, 4, G, G]
            delta_gt_rad = action_gt.squeeze(1)                          # [B, 4]
            if use_sincos:
                target = _delta_to_sincos(delta_gt_rad)                  # [B, 8]
            else:
                target = delta_gt_rad

            # 1. Prediction loss
            pred_loss = criterion(raw_out, target)

            # 2. Sparsity: each region mask should be concentrated (low entropy)
            #    H = -Σ p·log(p), high entropy = uniform = bad
            eps = 1e-8
            entropy = -(masks * torch.log(masks + eps)).sum(dim=(-2, -1)).mean()
            # Encouraging LOW entropy (1e-3 weight — very light)
            sparsity_loss = -0.001 * entropy

            # 3. Diversity: different regions should focus on different cells
            K = masks.size(1)  # num_regions
            masks_flat = masks.reshape(masks.size(0), K, -1)
            overlap = torch.bmm(masks_flat, masks_flat.transpose(1, 2))
            eye = torch.eye(K, device=masks.device).unsqueeze(0)
            off_diag = overlap * (1 - eye)
            diversity_loss = 0.01 * off_diag.sum(dim=(-2, -1)).mean()

            # 4. Temporal smoothness
            temp_loss = 0.0  # simplified: avg_masks already mean over T

            loss = pred_loss + sparsity_loss + diversity_loss + temp_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        delta_pred_rad = model.decode_delta(raw_out.detach())   # [B, 4] rad
        mae = (delta_pred_rad - delta_gt_rad).abs().mean(dim=0).cpu().numpy()
        total_mae += mae
        n_batches += 1

        if step % config.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "mae": f"{mae.mean():.4f}"})

    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0.0
    total_mae = np.zeros(4)
    n_batches = 0
    use_sincos = getattr(model, 'use_sincos_output', False)

    for batch in tqdm(dataloader, desc="Validating"):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)

        raw_out, _ = model(rgb, elevation, qpos, excavator_id)
        delta_gt_rad = action_gt.squeeze(1)
        if use_sincos:
            target = _delta_to_sincos(delta_gt_rad)
        else:
            target = delta_gt_rad
        loss = criterion(raw_out, target)

        total_loss += loss.item()
        delta_pred_rad = model.decode_delta(raw_out)
        mae = (delta_pred_rad - delta_gt_rad).abs().mean(dim=0).cpu().numpy()
        total_mae += mae
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
    }


def save_checkpoint(model, optimizer, scaler, epoch, metrics, config, is_best=False):
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "best" if is_best else f"epoch_{epoch+1}"
    path = os.path.join(config.output_dir, f"yolo_checkpoint_{suffix}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)
    print(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ExcavatorVLA-YOLO")
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
    args = parser.parse_args()

    config = Config()

    # ── CLI overrides ──
    for key in ("data_dir", "epochs", "batch_size", "lr", "seq_len",
                "sample_ratio", "img_size", "output_dir"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}, indent=2)}")

    # ── Datasets ──
    train_dataset = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="train", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )
    val_dataset = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="val", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True) if len(val_dataset) > 0 else None

    # ── Model ──
    model = ExcavatorVLAYolo(
        seq_len=config.seq_len, img_size=config.img_size,
        hidden_dim=config.hidden_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ff_dim=config.ff_dim,
        dropout=config.dropout, pretrained=config.pretrained,
    ).to(config.device)

    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    G = config.img_size // 16
    print(f"  Grid size: {G}×{G}")
    print(f"  Tokens per sequence: {config.seq_len} × {G}² grid + {model.num_queries} queries = "
          f"{config.seq_len * G ** 2 + model.num_queries}")

    # ── Optimiser + scheduler ──
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler   = GradScaler()
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

        # Filter: only load keys that match in both name AND shape
        # (allows loading backbone-only pretrained checkpoints with different head dims)
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

        # Only load optimizer/scaler if this is a full training checkpoint (not backbone-only)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scaler,
                                     criterion, config, epoch)
        scheduler.step()

        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(config.ema_decay).add_(p.data, alpha=1 - config.ema_decay)

        val_metrics = {"loss": float("nan"), "mae": [0, 0, 0, 0], "mae_mean": float("nan")}
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae_mean"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae_mean"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Train Loss: {train_metrics['loss']:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Train MAE: {train_metrics['mae_mean']:.4f} | "
              f"Val MAE: {val_metrics['mae_mean']:.4f} | "
              f"Time: {elapsed:.0f}s")
        print(f"  Per-joint MAE - "
              f"Train: {[f'{x:.4f}' for x in train_metrics['mae']]} | "
              f"Val:   {[f'{x:.4f}' for x in val_metrics['mae']]}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config, is_best=True)

        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config)

    save_checkpoint(model, optimizer, scaler, config.epochs - 1, val_metrics, config)
    print(f"\nTraining complete.  Best val loss: {best_loss:.6f}")

    with open(os.path.join(config.output_dir, "yolo_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
