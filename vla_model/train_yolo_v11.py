"""V11 training entrypoint: pure-visual frame residual model.

V11 retains V10's training-only pose auxiliary target, but never passes qpos
into the model.  The learning-rate scheduler advances once per optimizer batch.
"""
import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.config import Config
from vla_model.dataset import ExcavatorDataset
from vla_model.metrics import grouped_regression_metrics
from vla_model.model_yolo import ExcavatorVLAYolo, count_parameters
from vla_model.train_yolo_v10 import _compute_r2, _rad_to_output


Config.v11_pose_aux_weight = 0.05


def build_v11_model(seq_len=8, img_size=224, hidden_dim=512, n_heads=8,
                    n_layers=4, ff_dim=2048, dropout=0.1, pretrained=True,
                    num_excavators=4):
    """Construct the V11 pure-visual frame-residual architecture."""
    return ExcavatorVLAYolo(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim, dropout=dropout,
        pretrained=pretrained, num_excavators=num_excavators, version="v11",
    )


def build_batch_scheduler(optimizer, total_steps, warmup_ratio):
    """Build a scheduler whose iteration unit is one optimizer update."""
    total_steps = max(1, int(total_steps))
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps - 1)
    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def step_batch_scheduler(scheduler):
    """Advance learning rate exactly once after each optimizer update."""
    scheduler.step()


def _losses(model, raw_out, masks_spatial, pose_aux, action_gt, qpos, criterion, aux_weight):
    action_gt_rad = action_gt.squeeze(1)
    target = _rad_to_output(action_gt_rad, getattr(model, "out_dims", (2, 2, 2, 2)))
    pred_loss = criterion(raw_out, target)
    raw_4d = raw_out.view(raw_out.size(0), 4, 2)
    circle_loss = 0.1 * (raw_4d.pow(2).sum(dim=-1) - 1.0).pow(2).mean()

    area_mean = masks_spatial.mean(dim=(-2, -1)).mean()
    area_loss = (torch.relu(0.05 - area_mean).pow(2) +
                 torch.relu(area_mean - 0.30).pow(2))
    joint_count = masks_spatial.size(1)
    mask_flat = masks_spatial.reshape(masks_spatial.size(0), joint_count, -1)
    mask_norm = mask_flat / mask_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = torch.bmm(mask_norm, mask_norm.transpose(1, 2))
    off_diag = cosine * (1 - torch.eye(joint_count, device=cosine.device).unsqueeze(0))
    diversity_loss = 0.5 * torch.relu(off_diag - 0.3).pow(2).sum(dim=(-2, -1)).mean()
    temporal_loss = 0.005 * (masks_spatial[:, :, 1:] - masks_spatial[:, :, :-1]).abs().mean()
    pose_aux_loss = criterion(pose_aux, qpos[:, -1])
    return (pred_loss + circle_loss + area_loss + diversity_loss + temporal_loss +
            aux_weight * pose_aux_loss), action_gt_rad


def train_epoch(model, dataloader, optimizer, scheduler, scaler, criterion, config, epoch):
    model.train()
    sums = {"loss": 0.0, "mae": np.zeros(4), "ss_res": np.zeros(4),
            "sum_y": np.zeros(4), "sum_y2": np.zeros(4), "count": 0, "batches": 0}
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch + 1}")
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)
        optimizer.zero_grad()
        with autocast(enabled=config.device == "cuda"):
            # qpos is intentionally absent from forward: it is only an auxiliary label.
            raw_out, _, masks_spatial, pose_aux = model(
                rgb, elevation, None, excavator_id, return_aux=True
            )
            loss, action_gt_rad = _losses(
                model, raw_out, masks_spatial, pose_aux, action_gt, qpos, criterion,
                getattr(config, "v11_pose_aux_weight", 0.05),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step_batch_scheduler(scheduler)

        pred = model.decode_action(raw_out.detach())
        mae = (pred - action_gt_rad).abs().mean(dim=0).cpu().numpy()
        pred_np, label_np = pred.cpu().numpy(), action_gt_rad.cpu().numpy()
        sums["loss"] += loss.item()
        sums["mae"] += mae
        sums["ss_res"] += ((pred_np - label_np) ** 2).sum(axis=0)
        sums["sum_y"] += label_np.sum(axis=0)
        sums["sum_y2"] += (label_np ** 2).sum(axis=0)
        sums["count"] += len(label_np)
        sums["batches"] += 1
        if step % config.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}", mae=f"{mae.mean():.4f}")
    r2, r2_mean = _compute_r2(sums["ss_res"], sums["sum_y"], sums["sum_y2"], sums["count"])
    return {"loss": sums["loss"] / sums["batches"], "mae": (sums["mae"] / sums["batches"]).tolist(),
            "mae_mean": float(sums["mae"].mean() / sums["batches"]), "r2": r2.tolist(), "r2_mean": r2_mean}


@torch.no_grad()
def validate(model, dataloader, criterion, config):
    """Validate V11 as a pure-visual model; qpos never enters forward."""
    model.eval()
    total_loss, total_mae = 0.0, np.zeros(4)
    ss_res, sum_y, sum_y2 = np.zeros(4), np.zeros(4), np.zeros(4)
    n_total = n_batches = 0
    predictions, targets, excavator_ids, episode_ids = [], [], [], []
    for batch in tqdm(dataloader, desc="Validating"):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt_rad = batch["action"].to(config.device).squeeze(1)
        raw_out, _, _ = model(rgb, elevation, None, excavator_id)
        target = _rad_to_output(action_gt_rad, getattr(model, "out_dims", (2, 2, 2, 2)))
        total_loss += criterion(raw_out, target).item()
        pred = model.decode_action(raw_out)
        mae = (pred - action_gt_rad).abs().mean(dim=0).cpu().numpy()
        pred_np, label_np = pred.cpu().numpy(), action_gt_rad.cpu().numpy()
        total_mae += mae
        ss_res += ((pred_np - label_np) ** 2).sum(axis=0)
        sum_y += label_np.sum(axis=0)
        sum_y2 += (label_np ** 2).sum(axis=0)
        predictions.append(pred_np)
        targets.append(label_np)
        excavator_ids.append(excavator_id.detach().cpu().numpy())
        # New datasets provide local episode provenance.  The fallback keeps
        # validation usable for custom legacy datasets.
        episode = batch.get("episode_id")
        if episode is None:
            episode = torch.full_like(excavator_id, -1)
        episode_ids.append(episode.detach().cpu().numpy())
        n_total += len(label_np)
        n_batches += 1
    r2, r2_mean = _compute_r2(ss_res, sum_y, sum_y2, n_total)
    grouped = grouped_regression_metrics(
        np.concatenate(predictions), np.concatenate(targets),
        np.concatenate(excavator_ids), np.concatenate(episode_ids),
    )
    return {"loss": total_loss / n_batches, "mae": (total_mae / n_batches).tolist(),
            "mae_mean": float(total_mae.mean() / n_batches), "r2": r2.tolist(), "r2_mean": r2_mean,
            **grouped}


def save_checkpoint(model, optimizer, scaler, epoch, metrics, config, is_best=False):
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "best" if is_best else f"epoch_{epoch + 1}"
    path = os.path.join(config.output_dir, f"yolo_checkpoint_{suffix}.pt")
    torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "metrics": metrics, "config": config, "model_version": "v11"}, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ExcavatorVLA-YOLO V11")
    for name, kind in (("data_dir", str), ("epochs", int), ("batch_size", int), ("lr", float),
                       ("seq_len", int), ("sample_ratio", float), ("img_size", int), ("output_dir", str)):
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--pose_aux_weight", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    args = parser.parse_args()
    config = Config()
    for key in ("data_dir", "epochs", "batch_size", "lr", "seq_len", "sample_ratio", "img_size", "output_dir"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    config.v11_pose_aux_weight = args.pose_aux_weight
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}; V11 pose_aux_weight: {config.v11_pose_aux_weight}")
    print(json.dumps({k: str(v) for k, v in config.__dict__.items() if not k.startswith("_")}, indent=2))

    split = 1.0 if args.overfit else config.train_split
    train_data = ExcavatorDataset(config.data_dir, config.seq_len, config.action_chunk, config.img_size,
                                  "train", split, config.sample_ratio)
    val_data = ExcavatorDataset(config.data_dir, config.seq_len, config.action_chunk, config.img_size,
                                "val", split, config.sample_ratio)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True) if len(val_data) else None
    model = build_v11_model(config.seq_len, config.img_size, config.hidden_dim, config.n_heads,
                            config.n_layers, config.ff_dim, config.dropout, config.pretrained).to(config.device)
    print(f"Model parameters: {count_parameters(model)}")
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_batch_scheduler(optimizer, len(train_loader) * config.epochs, config.warmup_ratio)
    scaler, criterion = GradScaler(), nn.MSELoss()
    ema_model = deepcopy(model).eval() if config.use_ema else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad = False

    start_epoch, best_loss, stale_epochs = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
        target_state = model.state_dict()
        compatible = {k: v for k, v in checkpoint["model_state_dict"].items()
                      if k in target_state and target_state[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        start_epoch, best_loss = checkpoint.get("epoch", 0), checkpoint.get("metrics", {}).get("loss", float("inf"))
        print(f"[resume] loaded {len(compatible)} compatible keys from {args.resume}")

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "train_r2": [], "val_r2": []}
    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, config, epoch)
        if ema_model is not None:
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.data.mul_(config.ema_decay).add_(parameter.data, alpha=1 - config.ema_decay)
        val_metrics = validate(model, val_loader, criterion, config) if val_loader is not None else train_metrics
        for key, metrics_key in (("train_loss", "loss"), ("val_loss", "loss"), ("train_mae", "mae_mean"),
                                 ("val_mae", "mae_mean"), ("train_r2", "r2_mean"), ("val_r2", "r2_mean")):
            history[key].append((train_metrics if key.startswith("train") else val_metrics)[metrics_key])
        print(f"Epoch {epoch + 1}/{config.epochs} LR={scheduler.get_last_lr()[0]:.2e} "
              f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} time={time.time() - started:.0f}s")
        if "overall" in val_metrics:
            print(f"  Validation groups overall: MAE={val_metrics['overall']['mae_mean']:.4f}, "
                  f"R2={val_metrics['overall']['r2_mean']:.4f}")
            for excavator_id, metrics in val_metrics["by_excavator"].items():
                print(f"    Excavator {excavator_id}: n={metrics['n_samples']}, "
                      f"MAE={metrics['mae_mean']:.4f}, R2={metrics['r2_mean']:.4f}")
            for episode_key, metrics in val_metrics["by_episode"].items():
                print(f"    Episode {episode_key}: n={metrics['n_samples']}, "
                      f"MAE={metrics['mae_mean']:.4f}, R2={metrics['r2_mean']:.4f}")
        if val_metrics["loss"] < best_loss:
            best_loss, stale_epochs = val_metrics["loss"], 0
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config, is_best=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
                break
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config)
    save_checkpoint(model, optimizer, scaler, epoch, val_metrics, config)
    with open(os.path.join(config.output_dir, "yolo_history.json"), "w") as history_file:
        json.dump(history, history_file, indent=2)


if __name__ == "__main__":
    main()
