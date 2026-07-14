"""Excavator-STVTA V12 training entrypoint.

Dual-branch RGB/Elevation model with per-joint modality fusion.
Training includes modality dropout (10% chance of suppressing one branch).
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
from vla_model.model_stvta import ExcavatorSTVTA, count_parameters


Config.v12_pose_aux_weight = 0.05


def _compute_r2(ss_res, sum_y, sum_y2, n):
    ss_tot = sum_y2 - (sum_y ** 2) / n
    ss_tot = np.maximum(ss_tot, 1e-10)
    r2 = 1 - ss_res / ss_tot
    return r2, float(r2.mean())


def _rad_to_output(rad):
    """[B, 4] rad → [B, 8] sin/cos."""
    sin, cos = torch.sin(rad), torch.cos(rad)
    return torch.stack([sin, cos], dim=-1).reshape(rad.size(0), -1)


def build_v12_model(seq_len=8, img_size=224, hidden_dim=512, n_heads=8,
                     n_layers=4, ff_dim=2048, dropout=0.1, pretrained=True,
                     num_excavators=4):
    return ExcavatorSTVTA(
        seq_len=seq_len, img_size=img_size, hidden_dim=hidden_dim,
        n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
        dropout=dropout, pretrained=pretrained, num_excavators=num_excavators,
    )


def _stvta_losses(model, raw_out, masks_spatial, pose_aux, action_gt, qpos, criterion, aux_weight):
    action_gt_rad = action_gt.squeeze(1)
    target = _rad_to_output(action_gt_rad)
    pred_loss = criterion(raw_out, target)
    raw_4d = raw_out.view(raw_out.size(0), 4, 2)
    circle_loss = 0.1 * (raw_4d.pow(2).sum(dim=-1) - 1.0).pow(2).mean()
    area_mean = masks_spatial.mean(dim=(-2, -1)).mean()
    area_loss = (torch.relu(0.05 - area_mean).pow(2) +
                 torch.relu(area_mean - 0.30).pow(2))
    # V12.2: diversity_loss disabled, temporal_loss reduced
    diversity_loss = 0.0
    temporal_loss = 0.0005 * (masks_spatial[:, :, :, 1:] -
                               masks_spatial[:, :, :, :-1]).abs().mean()
    pose_aux_loss = criterion(pose_aux, qpos[:, -1])
    mask_spatial_std = masks_spatial.std(dim=(-2, -1)).mean()
    mask_temporal_std = (masks_spatial[:, :, :, 1:] - masks_spatial[:, :, :, :-1]).abs().mean()
    return (pred_loss + circle_loss + area_loss + temporal_loss +
            aux_weight * pose_aux_loss), action_gt_rad, {
                "mask_spatial_std": mask_spatial_std.item(),
                "mask_temporal_std": mask_temporal_std.item()}


def train_epoch(model, dataloader, optimizer, scheduler, scaler, criterion, config, epoch):
    model.train()
    sums = {"loss": 0.0, "mae": np.zeros(4), "ss_res": np.zeros(4),
            "sum_y": np.zeros(4), "sum_y2": np.zeros(4), "count": 0, "batches": 0}
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch + 1}")
    aux_weight = getattr(config, "v12_pose_aux_weight", 0.05)
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        qpos = batch["qpos"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt = batch["action"].to(config.device)
        optimizer.zero_grad()
        with autocast(enabled=config.device == "cuda"):
            raw_out, _, masks_spatial, _, pose_aux = model(
                rgb, elevation, excavator_id=excavator_id, return_aux=True,
            )
            loss, action_gt_rad, mask_stats = _stvta_losses(
                model, raw_out, masks_spatial, pose_aux, action_gt, qpos, criterion, aux_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

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
            pbar.set_postfix(loss=f"{loss.item():.5f}", mae=f"{mae.mean():.4f}", s_std=f"{mask_stats['mask_spatial_std']:.4f}", t_std=f"{mask_stats['mask_temporal_std']:.4f}")
    r2, r2_mean = _compute_r2(sums["ss_res"], sums["sum_y"], sums["sum_y2"], sums["count"])
    return {"loss": sums["loss"] / sums["batches"],
            "mae": (sums["mae"] / sums["batches"]).tolist(),
            "mae_mean": float(sums["mae"].mean() / sums["batches"]),
            "r2": r2.tolist(), "r2_mean": r2_mean}


@torch.no_grad()
def validate(model, dataloader, criterion, config):
    model.eval()
    total_loss, total_mae = 0.0, np.zeros(4)
    ss_res, sum_y, sum_y2 = np.zeros(4), np.zeros(4), np.zeros(4)
    n_total = n_batches = 0
    all_preds, all_labels, all_excv = [], [], []
    for batch in tqdm(dataloader, desc="Validating"):
        rgb = batch["rgb"].to(config.device)
        elevation = batch["elevation"].to(config.device)
        excavator_id = batch["excavator_id"].to(config.device)
        action_gt_rad = batch["action"].to(config.device).squeeze(1)
        raw_out, _, _, _ = model(rgb, elevation, excavator_id=excavator_id)
        target = _rad_to_output(action_gt_rad)
        total_loss += criterion(raw_out, target).item()
        pred = model.decode_action(raw_out)
        mae = (pred - action_gt_rad).abs().mean(dim=0).cpu().numpy()
        pred_np, label_np = pred.cpu().numpy(), action_gt_rad.cpu().numpy()
        total_mae += mae
        ss_res += ((pred_np - label_np) ** 2).sum(axis=0)
        sum_y += label_np.sum(axis=0)
        sum_y2 += (label_np ** 2).sum(axis=0)
        n_total += len(label_np)
        n_batches += 1
        all_preds.append(pred_np); all_labels.append(label_np)
        all_excv.append(excavator_id.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_excv = np.concatenate(all_excv)
    r2, r2_mean = _compute_r2(ss_res, sum_y, sum_y2, n_total)

    # Per-excavator grouped metrics
    grouped = {}
    for eid in sorted(set(all_excv)):
        mask = all_excv == eid
        pred_e, label_e = all_preds[mask], all_labels[mask]
        mae_e = np.abs(pred_e - label_e).mean(axis=0)
        ss_res_e = ((pred_e - label_e) ** 2).sum(axis=0)
        sum_y_e = label_e.sum(axis=0)
        sum_y2_e = (label_e ** 2).sum(axis=0)
        r2_e, r2_mean_e = _compute_r2(ss_res_e, sum_y_e, sum_y2_e, len(label_e))
        grouped[eid] = {"n": len(label_e), "mae": mae_e.tolist(), "mae_mean": float(mae_e.mean()),
                        "r2": r2_e.tolist(), "r2_mean": r2_mean_e}
    return {"loss": total_loss / n_batches,
            "mae": (total_mae / n_batches).tolist(),
            "mae_mean": float(total_mae.mean() / n_batches),
            "r2": r2.tolist(), "r2_mean": r2_mean}, grouped


def save_checkpoint(model, optimizer, scaler, epoch, metrics, config, is_best=False):
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "best" if is_best else f"epoch_{epoch+1}"
    path = os.path.join(config.output_dir, f"stvta_checkpoint_{suffix}.pt")
    torch.save({
        "epoch": epoch + 1, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics, "model_version": "v12",
    }, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Excavator-STVTA V12")
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
    for key in ("data_dir", "epochs", "batch_size", "lr", "seq_len",
                "sample_ratio", "img_size", "output_dir"):
        val = getattr(args, key, None)
        if val is not None: setattr(config, key, val)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config.device}")
    print(f"V12 modality dropout: 10%")

    # ── Datasets ──
    train_ds = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="train", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )
    val_ds = ExcavatorDataset(
        data_dir=config.data_dir, seq_len=config.seq_len,
        action_chunk=config.action_chunk, img_size=config.img_size,
        split="val", train_split=config.train_split if not args.overfit else 1.0,
        sample_ratio=config.sample_ratio,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True) if len(val_ds) > 0 else None

    # ── Model ──
    model = build_v12_model(
        seq_len=config.seq_len, img_size=config.img_size,
        hidden_dim=config.hidden_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ff_dim=config.ff_dim,
        dropout=config.dropout, pretrained=config.pretrained,
    ).to(config.device)

    params = count_parameters(model)
    G = config.img_size // 16
    print(f"Model: {params['total']:,} params | grid={G}x{G} | tokens={config.seq_len*G*G}")

    # ── Optimiser ──
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = GradScaler()
    criterion = nn.MSELoss()

    # ── Resume ──
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=config.device, weights_only=False)
        resume_state = ckpt["model_state_dict"]
        model_state = model.state_dict()
        filtered, skipped = {}, 0
        for k, v in resume_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else: skipped += 1
        model.load_state_dict(filtered, strict=False)
        print(f"  [resume] {len(filtered)} keys loaded, {skipped} skipped")
        if "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [],
               "train_r2": [], "val_r2": []}
    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, config, epoch)
        val_m = {"loss": float("nan"), "mae": [0,0,0,0], "mae_mean": float("nan"),
                 "r2": [0,0,0,0], "r2_mean": float("nan")}
        val_grouped = {}
        if val_loader is not None and len(val_loader) > 0:
            val_m, val_grouped = validate(model, val_loader, criterion, config)
        for k, v in [("train_loss", train_m["loss"]), ("val_loss", val_m["loss"]),
                     ("train_mae", train_m["mae_mean"]), ("val_mae", val_m["mae_mean"]),
                     ("train_r2", train_m["r2_mean"]), ("val_r2", val_m["r2_mean"])]:
            history[k].append(v)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{config.epochs} | LR={scheduler.get_last_lr()[0]:.2e} | "
              f"T Loss={train_m['loss']:.6f} V Loss={val_m['loss']:.6f} | "
              f"T MAE={train_m['mae_mean']:.4f} V MAE={val_m['mae_mean']:.4f} | "
              f"T R²={train_m['r2_mean']:.4f} V R²={val_m['r2_mean']:.4f} | {elapsed:.0f}s")
        print(f"  Per-joint MAE T: {[f'{x:.4f}' for x in train_m['mae']]} | V: {[f'{x:.4f}' for x in val_m['mae']]}")
        print(f"  Per-joint R²  T: {[f'{x:.4f}' for x in train_m['r2']]} | V: {[f'{x:.4f}' for x in val_m['r2']]}")
        for eid in sorted(val_grouped.keys()):
            g = val_grouped[eid]
            print(f"  Excavator {eid}: n={g['n']}, MAE={g['mae_mean']:.4f}, R²={g['r2_mean']:.4f}")
        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            save_checkpoint(model, optimizer, scaler, epoch, val_m, config, is_best=True)
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_m, config)
    save_checkpoint(model, optimizer, scaler, config.epochs - 1, val_m, config)
    with open(os.path.join(config.output_dir, "stvta_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
