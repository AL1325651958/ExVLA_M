"""Benchmark all trained model versions across excavator types.

Walks output/ directories, finds every .pt checkpoint, loads it, runs inference
on all excavator episodes (75/306/490), computes per-joint MAE/R² (circular for Swing),
and exports results to benchmark_results.csv.

Usage:
  python vla_model/benchmark_all.py --output_dir output --data_dir data/excavator-motion --device cuda
  python vla_model/benchmark_all.py --output_dir output --data_dir data/excavator-motion --device cuda --best_only
"""

import os
import sys
import csv
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model_yolo import (
    ExcavatorVLAYolo,
    load_compatible_state_dict,
    upgrade_legacy_v17_1_state_dict,
)
from vla_model.model_stvta import ExcavatorSTVTA
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD

import cv2

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
EXCV_NAMES = {0: '75', 1: '306', 2: '490', 3: 'unknown'}


# ── Utilities ──

def _circular_error(pred_rad, gt_rad):
    """Wrap-aware difference in [-π, π]."""
    delta = pred_rad - gt_rad
    return np.arctan2(np.sin(delta), np.cos(delta))


def _compute_per_joint_metrics(predictions, targets):
    """Compute per-joint MAE and R². Swing uses circular error."""
    valid = ~np.isnan(predictions[:, 0])
    pred = predictions[valid]
    gt = targets[valid]
    n = len(pred)

    # MAE
    mae = np.zeros(4)
    for j in range(3):
        mae[j] = np.abs(pred[:, j] - gt[:, j]).mean()
    mae[3] = np.abs(_circular_error(pred[:, 3], gt[:, 3])).mean()

    # R²
    r2 = np.zeros(4)
    for j in range(3):
        ss_res = ((pred[:, j] - gt[:, j]) ** 2).sum()
        ss_tot = ((gt[:, j] - gt[:, j].mean()) ** 2).sum()
        r2[j] = 1 - ss_res / max(ss_tot, 1e-10)
    # Swing: circular
    swing_err = _circular_error(pred[:, 3], gt[:, 3])
    ss_res_s = (swing_err ** 2).sum()
    s = gt[:, 3]
    mean_angle = np.arctan2(np.sin(s).mean(), np.cos(s).mean())
    centered = _circular_error(s, mean_angle)
    ss_tot_s = max((centered ** 2).sum(), 1e-10)
    r2[3] = 1 - ss_res_s / ss_tot_s

    return mae, r2, n


def preprocess_image(img_bgr, size=224):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


# ── Checkpoint discovery ──

def find_checkpoints(output_dir, best_only=False):
    """Find all checkpoint files. Returns list of (path, dir_name)."""
    all_ckpts = []
    pattern = "**/yolo_v17_1_checkpoint_best_swing.pt"
    for pt in glob.glob(os.path.join(output_dir, "**", "*.pt"), recursive=True):
        name = os.path.basename(pt)
        parent = os.path.basename(os.path.dirname(pt))
        # Skip non-checkpoint .pt files
        if any(skip in name.lower() for skip in ['history', 'pretrained', 'backbone']):
            continue
        if parent in ('checkpoints',) and 'pretrained' in name:
            continue
        all_ckpts.append((pt, parent))
    return sorted(all_ckpts, key=lambda x: x[1])


# ── Model loading ──

def detect_architecture(state_keys, ckpt_info):
    """Return ('yolo', version_arg) or ('stvta', version_str)."""
    # STVTA detection
    if any('rgb_branch' in k for k in state_keys):
        # Determine STVTA version
        has_joint_rel = any('joint_relation' in k or 'vel_aux_head' in k for k in state_keys)
        v = 'v13' if has_joint_rel else 'v12'
        return 'stvta', v

    # YOLO detection
    version_tag, _, version_arg = detect_yolo_version(state_keys, ckpt_info.get('model_version'))
    return 'yolo', version_arg


def detect_yolo_version(state_keys, checkpoint_version=None):
    """Detect YOLO version from state dict keys."""
    if str(checkpoint_version).lower() in ("v17.1", "v17", "v16"):
        return str(checkpoint_version).upper(), True, str(checkpoint_version).lower()

    # V17.3 / V17.1 model_version tags → v17.1
    if str(checkpoint_version).lower() in ("v17.3", "v17.1"):
        return "V17.1", True, "v17.1"

    has_v17 = any("joint_logit_bias" in k for k in state_keys)
    has_temporal = any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys)
    if has_v17 and has_temporal:
        return "V17.1", True, "v17.1"
    if has_v17:
        return "V17", True, "v17"

    has_v16 = any("rgb_proj" in k or "cross_rgb_from_elev" in k for k in state_keys)
    if has_v16:
        return "V16", True, "v16"

    has_v11 = any(k.startswith("motion_adapter.") for k in state_keys)
    if has_v11:
        return "V11", False, "v11"

    has_v10 = any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys)
    if has_v10:
        return "V10", False, "v10"
    return "V9", False, "v9"


def infer_yolo_config(state_dict):
    """Infer hidden_dim, n_layers, ff_dim from YOLO checkpoint."""
    attention = state_dict.get(
        "encoder.layers.0.self_attn.in_proj_weight",
        state_dict.get("encoder.layers.0.linear1.weight"),
    )
    if attention is None:
        raise KeyError("No encoder weights found")
    layer_indices = {
        int(k.split(".")[2]) for k in state_dict
        if k.startswith("encoder.layers.")
    }
    hidden_dim = attention.shape[1]
    n_layers = max(layer_indices) + 1 if layer_indices else 4
    ff_dim = state_dict.get("encoder.layers.0.linear1.weight", attention).shape[0]
    return hidden_dim, n_layers, ff_dim


def remap_legacy_keys(state_dict):
    """Apply old-checkpoint key remapping (V2-V7 → current key names)."""
    sd = dict(state_dict)
    sd_keys = set(sd.keys())
    remapped = {}

    # R1: skip delta_head and old-style action_head (V2-V4, single shared head)
    for k in list(sd.keys()):
        if "delta_head" in k:
            del sd[k]
        if "action_head." in k and "action_heads" not in k:
            del sd[k]
        if "qpos_mod" in k or "qpos_proj" in k:
            del sd[k]

    # R2: V5 old 6-joint → remap to 4-joint (skip 6. and 3., clone for j in 0..3)
    is_v5 = any("action_heads" in sk for sk in sd.keys()) and not any("joint_embed" in sk for sk in sd.keys())
    if is_v5:
        for k, val in list(sd.items()):
            if "action_heads." in k:
                parts = k.split(".")
                eid = int(parts[1])
                rest = ".".join(parts[2:])
                if rest.startswith("6.") or rest.startswith("3."):
                    del sd[k]
                    continue
                for j in range(4):
                    remapped[f"action_heads.{eid}.{j}.{rest}"] = val.clone()

    # R3: mask_head → mask_heads (V5-V7 shared mask_head)
    for k, val in list(sd.items()):
        if "mask_head" in k and "mask_heads" not in k:
            for j in range(4):
                remapped[k.replace("mask_head", f"mask_heads.{j}")] = val.clone()
            del sd[k]

    # R4: query_tokens → joint_queries
    if "query_tokens" in sd and "joint_queries" not in sd:
        sd["joint_queries"] = sd.pop("query_tokens")

    sd.update(remapped)
    return sd


def load_yolo_model(ckpt_path, device):
    """Load a YOLO model with correct architecture."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    # Apply old-checkpoint key remapping (V2-V7)
    sd = remap_legacy_keys(sd)
    sd_keys = set(sd.keys())

    version_tag, is_v16, version_arg = detect_yolo_version(
        sd_keys, ckpt.get("model_version")
    )

    # Upgrade legacy V17 masks
    if version_arg in ("v17", "v17.1"):
        sd = upgrade_legacy_v17_1_state_dict(sd)

    # Infer config
    try:
        hidden_dim, n_layers, ff_dim = infer_yolo_config(sd)
    except KeyError:
        hidden_dim, n_layers, ff_dim = 512, 4, 2048

    model = ExcavatorVLAYolo(
        seq_len=8, img_size=224, hidden_dim=hidden_dim, n_heads=8,
        n_layers=n_layers, ff_dim=ff_dim, dropout=0.0, pretrained=False,
        version=version_arg,
    ).to(device)
    load_compatible_state_dict(model, sd)
    model.eval()
    return model, version_tag, ckpt.get("epoch", "?")


def load_stvta_model(ckpt_path, device):
    """Load an STVTA model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd_keys = set(sd.keys())

    # Infer config
    hidden_dim = sd.get("rgb_branch.grid_proj.weight").shape[0]
    n_layers = 4
    for k in sd_keys:
        if k.startswith("rgb_branch.encoder.layers."):
            idx = int(k.split(".")[3])
            n_layers = max(n_layers, idx + 1)
    ff_dim = sd.get("rgb_branch.encoder.layers.0.linear1.weight").shape[0]

    model = ExcavatorSTVTA(
        seq_len=8, img_size=224, hidden_dim=hidden_dim, n_heads=8,
        n_layers=n_layers, ff_dim=ff_dim, dropout=0.0, pretrained=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    is_v13 = any("joint_relation" in k or "vel_aux_head" in k for k in sd_keys)
    return model, f"STVTA_{'V13' if is_v13 else 'V12'}", ckpt.get("epoch", "?")


# ── Data loading ──

def load_episode(h5_path):
    """Load one HDF5 episode, return preprocessed arrays + metadata."""
    with h5py.File(h5_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        targets = np.zeros_like(qpos)
        targets[:-1] = qpos[1:]
        targets[-1] = qpos[-1]

    N = len(targets)
    rgb_pp = np.zeros((N, 3, 224, 224), dtype=np.float32)
    elev_pp = np.zeros((N, 3, 224, 224), dtype=np.float32)
    for i in range(N):
        rgb_pp[i] = preprocess_image(mains[i])
        elev_pp[i] = preprocess_image(elevations[i])

    # Parse excavator ID
    pl = h5_path.lower()
    if '/75/' in pl or '\\75\\' in pl:
        excv_id = 0
    elif '/306/' in pl or '\\306\\' in pl:
        excv_id = 1
    elif '/490/' in pl or '\\490\\' in pl:
        excv_id = 2
    else:
        excv_id = 3

    return rgb_pp, elev_pp, qpos, targets, excv_id, N


def find_episodes(data_dir):
    """Find all .h5/.hdf5 episode files grouped by excavator type."""
    episodes = defaultdict(list)

    for fp in sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True)):
        excv_id = _parse_excv_from_path(fp)
        episodes[excv_id].append(fp)

    for fp in sorted(glob.glob(os.path.join(data_dir, "**", "*.hdf5"), recursive=True)):
        excv_id = _parse_excv_from_path(fp)
        episodes[excv_id].append(fp)

    return episodes


def _parse_excv_from_path(fp):
    pl = fp.lower()
    if '/75/' in pl or '\\75\\' in pl:
        return 0
    if '/306/' in pl or '\\306\\' in pl:
        return 1
    if '/490/' in pl or '\\490\\' in pl:
        return 2
    return 3


# ── Inference ──

@torch.no_grad()
def run_inference_yolo(model, rgb_pp, elev_pp, excv_id, device, sample_ratio=1.0):
    """Run YOLO inference over an episode."""
    N = len(rgb_pp)
    T = 8
    excv_t = torch.tensor([excv_id], dtype=torch.long).to(device)

    predictions = np.full((N, 4), np.nan, dtype=np.float32)
    predictions[:T - 1] = np.zeros((T - 1, 4))  # dummy fill

    step = max(1, int(1.0 / sample_ratio)) if sample_ratio < 1.0 else 1
    for start in range(0, N - T, step):
        end = start + T
        rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
        elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        action, _, _ = model(rgb_seq, elev_seq, None, excv_t)
        predictions[start + T - 1] = model.decode_action(action)[0].cpu().numpy()

    return predictions


@torch.no_grad()
def run_inference_stvta(model, rgb_pp, elev_pp, excv_id, device, sample_ratio=1.0):
    """Run STVTA inference over an episode."""
    N = len(rgb_pp)
    T = 8
    excv_t = torch.tensor([excv_id], dtype=torch.long).to(device)

    predictions = np.full((N, 4), np.nan, dtype=np.float32)
    predictions[:T - 1] = np.zeros((T - 1, 4))

    step = max(1, int(1.0 / sample_ratio)) if sample_ratio < 1.0 else 1
    for start in range(0, N - T, step):
        end = start + T
        rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
        elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        outputs = model(rgb_seq, elev_seq, excavator_id=excv_t)
        action = outputs[0]
        predictions[start + T - 1] = model.decode_action(action)[0].cpu().numpy()

    return predictions


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Benchmark all model versions")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="data/excavator-motion")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--best_only", action="store_true",
                        help="Only benchmark 'best' checkpoints, skip epoch_XX ones")
    parser.add_argument("--skip_dirs", type=str, nargs="*", default=[],
                        help="Output directories to skip")
    parser.add_argument("--episodes_per_excv", type=int, default=0,
                        help="Max episodes per excavator (0=all)")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="Fraction of windows to evaluate per episode (default 0.1)")
    parser.add_argument("--exclude_306", action="store_true",
                        help="Skip excavator 306 (nighttime data)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  sample_ratio={args.sample_ratio}")

    # ── Find episodes ──
    episodes = find_episodes(args.data_dir)
    if args.exclude_306:
        episodes.pop(1, None)  # remove excavator 306
    for eid in sorted(episodes.keys()):
        print(f"  Excavator {EXCV_NAMES[eid]}: {len(episodes[eid])} episodes")
        if args.episodes_per_excv > 0:
            episodes[eid] = episodes[eid][:args.episodes_per_excv]

    # ── Pre-load all episodes ──
    print("\nPre-loading episodes...")
    episode_data = {}  # (excv_id, ep_idx) -> (rgb_pp, elev_pp, targets)
    for eid in sorted(episodes.keys()):
        for ep_idx, fp in enumerate(tqdm(episodes[eid], desc=f"  Excavator {EXCV_NAMES[eid]}")):
            rgb_pp, elev_pp, qpos, targets, _, N = load_episode(fp)
            episode_data[(eid, ep_idx)] = (rgb_pp, elev_pp, targets, N)

    # ── Find checkpoints ──
    print("\nFinding checkpoints...")
    ckpts = find_checkpoints(args.output_dir, args.best_only)

    # Filter: prefer best checkpoints, skip epoch_XX unless --best_only is not set
    if args.best_only:
        ckpts = [(p, d) for p, d in ckpts if 'best' in os.path.basename(p).lower()]
    else:
        # Keep best + periodic epoch checkpoints, but de-prioritize epoch ones by sorting
        ckpts.sort(key=lambda x: (
            x[1],  # sort by dir first
            0 if 'best' in os.path.basename(x[0]).lower() else 1,  # best first
            x[0]
        ))

    if args.skip_dirs:
        ckpts = [(p, d) for p, d in ckpts if d not in args.skip_dirs]

    print(f"  Found {len(ckpts)} checkpoints to benchmark")

    # ── Benchmark ──
    results = []

    for ckpt_idx, (ckpt_path, dir_name) in enumerate(ckpts):
        ckpt_name = os.path.basename(ckpt_path).replace('.pt', '')
        print(f"\n[{ckpt_idx + 1}/{len(ckpts)}] {dir_name}/{ckpt_name}")

        # Load model
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            sd_keys = set(sd.keys())
            arch, version_arg = detect_architecture(sd_keys, ckpt)

            if arch == 'yolo':
                model, version_tag, epoch = load_yolo_model(ckpt_path, device)
                run_fn = run_inference_yolo
            else:
                model, version_tag, epoch = load_stvta_model(ckpt_path, device)
                run_fn = run_inference_stvta

            version_label = f"{version_tag}" if version_tag else version_arg
            print(f"  {version_label} | epoch={epoch} | arch={arch}")

        except Exception as e:
            print(f"  SKIP: Failed to load model ({type(e).__name__}: {e})")
            continue

        # Run inference per excavator
        for eid in sorted(episodes.keys()):
            excv_name = EXCV_NAMES[eid]
            all_preds = []
            all_targets = []

            for ep_idx in range(len(episodes[eid])):
                key = (eid, ep_idx)
                if key not in episode_data:
                    continue
                rgb_pp, elev_pp, targets, N = episode_data[key]

                try:
                    preds = run_fn(model, rgb_pp, elev_pp, eid, device, sample_ratio=args.sample_ratio)
                    all_preds.append(preds)
                    all_targets.append(targets)
                except Exception as e:
                    print(f"    FAIL episode {ep_idx}: {e}")
                    continue

            if not all_preds:
                continue

            # Aggregate
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            mae, r2, n_valid = _compute_per_joint_metrics(all_preds, all_targets)

            row = {
                "checkpoint_dir": dir_name,
                "checkpoint_name": ckpt_name,
                "version": version_label,
                "architecture": arch,
                "epoch": epoch,
                "excavator": excv_name,
                "n_samples": n_valid,
                "mae_boom": mae[0], "mae_arm": mae[1],
                "mae_bucket": mae[2], "mae_swing": mae[3],
                "mae_mean": mae.mean(),
                "r2_boom": r2[0], "r2_arm": r2[1],
                "r2_bucket": r2[2], "r2_swing": r2[3],
                "r2_mean": r2.mean(),
            }
            results.append(row)

            print(f"  [{excv_name}] MAE: {mae[0]:.4f}/{mae[1]:.4f}/{mae[2]:.4f}/{mae[3]:.4f} | "
                  f"R²: {r2[0]:.4f}/{r2[1]:.4f}/{r2[2]:.4f}/{r2[3]:.4f} | mean MAE={mae.mean():.4f} mean R²={r2.mean():.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # ── Export CSV ──
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    if results:
        fieldnames = [
            "checkpoint_dir", "checkpoint_name", "version", "architecture", "epoch",
            "excavator", "n_samples",
            "mae_boom", "mae_arm", "mae_bucket", "mae_swing", "mae_mean",
            "r2_boom", "r2_arm", "r2_bucket", "r2_swing", "r2_mean",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} rows to {csv_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
