"""Training script for ExcavatorVLA model."""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.config import Config
from vla_model.model import ExcavatorVLA, count_parameters
from vla_model.dataset import ExcavatorDataset


def train_epoch(model, dataloader, optimizer, scaler, criterion, config, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = np.zeros(4)  # per-joint MAE
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)  # [B, K, 4]

        optimizer.zero_grad()

        with autocast():
            action_pred = model(rgb, elevation, qpos, excavator_id)  # [B, K, 4]
            mse_loss = criterion(action_pred, action_gt)

            # Temporal smoothness loss for action chunks (anti-jitter)
            if action_pred.size(1) > 1:
                smooth_loss = config.smooth_loss_weight * ((action_pred[:, 1:] - action_pred[:, :-1]) ** 2).mean()
            else:
                smooth_loss = 0.0

            loss = mse_loss + smooth_loss

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        mae = (action_pred.detach() - action_gt).abs().mean(dim=(0, 1)).cpu().numpy()
        total_mae += mae
        n_batches += 1

        if step % config.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "mae": f"{mae.mean():.4f}",
            })

    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, config):
    """Validation."""
    model.eval()
    total_loss = 0.0
    total_mae = np.zeros(4)
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)

        action_pred = model(rgb, elevation, qpos, excavator_id)
        loss = criterion(action_pred, action_gt)

        total_loss += loss.item()
        mae = (action_pred - action_gt).abs().mean(dim=(0, 1)).cpu().numpy()
        total_mae += mae
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mae": (total_mae / n_batches).tolist(),
        "mae_mean": float(total_mae.mean() / n_batches),
    }


def save_checkpoint(model, optimizer, scaler, epoch, metrics, config, is_best=False):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "best" if is_best else f"epoch_{epoch+1}"
    path = os.path.join(config.output_dir, f"checkpoint_{suffix}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ExcavatorVLA")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None, help="Fraction of data to use (0.2=20%% for fast training)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overfit", action="store_true", help="Overfit single episode")
    args = parser.parse_args()

    config = Config()

    # Override config with CLI args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.seq_len:
        config.seq_len = args.seq_len
    if args.sample_ratio is not None:
        config.sample_ratio = args.sample_ratio

    # Device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=2)}")

    # Datasets
    train_dataset = ExcavatorDataset(
        data_dir=config.data_dir,
        seq_len=config.seq_len,
        action_chunk=config.action_chunk,
        img_size=config.img_size,
        split="train",
        train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )
    val_dataset = ExcavatorDataset(
        data_dir=config.data_dir,
        seq_len=config.seq_len,
        action_chunk=config.action_chunk,
        img_size=config.img_size,
        split="val",
        train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None

    # Model
    model = ExcavatorVLA(
        seq_len=config.seq_len,
        action_chunk=config.action_chunk,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        drop_path_rate=config.drop_path_rate,
        pretrained=config.pretrained,
    ).to(config.device)

    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler = GradScaler()
    criterion = nn.MSELoss()

    # EMA (Exponential Moving Average) for better generalization
    ema_model = None
    if config.use_ema:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # Resume if specified
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["metrics"].get("loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, criterion, config, epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae_mean"])

        scheduler.step()

        # EMA update (smoothed weights for better generalization)
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(config.ema_decay).add_(p.data, alpha=1 - config.ema_decay)

        # Validate
        val_metrics = {"loss": float("nan"), "mae": [0, 0, 0, 0], "mae_mean": float("nan")}
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = validate(model, val_loader, criterion, config)
            history["val_loss"].append(val_metrics["loss"])
            history["val_mae"].append(val_metrics["mae_mean"])

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"LR: {lr_now:.2e} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Train MAE: {train_metrics['mae_mean']:.4f} | "
            f"Val MAE: {val_metrics['mae_mean']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        print(f"  Per-joint MAE - Train: {[f'{x:.4f}' for x in train_metrics['mae']]} | "
              f"Val: {[f'{x:.4f}' for x in val_metrics['mae']]}")

        # Save best
        is_best = val_metrics.get("loss", float("inf")) < best_loss
        if is_best:
            best_loss = val_metrics.get("loss", float("inf"))
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config, is_best=True)

        # Save periodic
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config)

    # Final save
    save_checkpoint(model, optimizer, scaler, config.epochs - 1, val_metrics, config, is_best=False)
    print(f"\nTraining complete. Best val loss: {best_loss:.6f}")

    # Save history
    with open(os.path.join(config.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
