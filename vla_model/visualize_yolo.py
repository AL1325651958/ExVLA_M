"""Visualize YOLO-ST-VLA: GT vs Pred curves + 4 learned task-region masks overlaid on RGB."""

import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import cv2
import imageio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model_yolo import ExcavatorVLAYolo
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
REGION_NAMES = ['Mask 1', 'Mask 2', 'Mask 3', 'Mask 4']
REGION_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]  # BGR

GT_COLOR = '#333333'
PRED_COLOR = '#e74c3c'

MAIN_W, MAIN_H = 270, 270
MASK_W, MASK_H = 270, 270
ELEV_W, ELEV_H = 270, 270
CURVE_W = 900
CURVE_H_PER_JOINT = 100
PAD = 6


def preprocess_image(img_bgr, size=224):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(img, (new_w, new_h))
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])


def render_masks(rgb_bgr, mask, color_idx, G):
    """Overlay a single region mask on the RGB image.
    mask: [G, G] numpy, values in [0,1]
    color_idx: 0-3 index into REGION_COLORS
    Returns: BGR image with heatmap overlay.
    """
    h, w = rgb_bgr.shape[:2]
    overlay = rgb_bgr.copy().astype(np.float32)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0, 1)
    m_min, m_max = mask.min(), mask.max()
    if m_max > m_min:
        mask = (mask - m_min) / (m_max - m_min)
    color = np.array(REGION_COLORS[color_idx], dtype=np.float32)
    alpha = 0.45
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha * mask) + color[c] * (alpha * mask)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_curves(timeline, targets, predictions, current_idx, frame_range=200):
    """Render 4 joint curves with GT and Pred."""
    N = len(timeline)
    half = frame_range // 2
    start = max(0, current_idx - half)
    end = min(N, current_idx + half)
    t_slice = timeline[start:end]
    x_min, x_max = t_slice[0], t_slice[-1]

    total_h = CURVE_H_PER_JOINT * 4
    fig, axes = plt.subplots(4, 1, figsize=(CURVE_W / 100, total_h / 100), dpi=100)
    fig.patch.set_facecolor('white')

    for j in range(4):
        ax = axes[j]
        ax.set_facecolor('white')
        ax.tick_params(labelsize=6)
        t_past = timeline[:current_idx + 1]
        gt_past = targets[:current_idx + 1, j]
        pred_past = predictions[:current_idx + 1, j]

        ax.plot(t_slice, targets[start:end, j], color='#cccccc', linewidth=0.6, alpha=0.7)
        ax.plot(t_past[start:], gt_past[start:], color=GT_COLOR, linewidth=1.0, label='GT')
        ax.plot(t_past[start:], pred_past[start:], color=PRED_COLOR, linewidth=1.0, label='Pred')
        ax.axvline(x=timeline[current_idx], color='#3498db', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.plot(timeline[current_idx], targets[current_idx, j], 'o', color=GT_COLOR, markersize=5)
        ax.plot(timeline[current_idx], predictions[current_idx, j], 'o', color=PRED_COLOR, markersize=5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=8, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.2, color='#999')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')
        if j == 0:
            ax.legend(loc='upper right', fontsize=7)
        if j < 3:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Frame', fontsize=8, color='#333')
    fig.tight_layout(pad=0.5, h_pad=0.2)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return buf


def select_mask_view(masks_spatial, mask_view):
    """Select mask view from temporal mask tensor.
    masks_spatial: [B, K, T, G, G]
    mask_view: 'avg' (V9 default) or 'last' (V10 default, raw last frame)
    """
    if mask_view == "last":
        return masks_spatial[:, :, -1]
    if mask_view == "avg":
        return masks_spatial.mean(dim=2)
    raise ValueError(f"Unknown mask view: {mask_view}")


def detect_model_version(state_keys, checkpoint):
    """Return the visualizer architecture version for a checkpoint.

    V11 is identified before V10 because it contains V10's temporal mixer as
    well as its own ``motion_adapter`` residual branch.  Older checkpoints
    intentionally retain the V9 fallback used by the existing remapping path.
    """
    checkpoint_version = checkpoint.get("model_version")
    if (checkpoint_version == "v11" or
            any(key.startswith("motion_adapter.") for key in state_keys)):
        return "v11"
    if (checkpoint_version == "v10" or
            any("temporal_mask_mixer" in key or "pose_aux_head" in key
                for key in state_keys)):
        return "v10"
    return "v9"


def run_visual_inference(model, rgb_seq, elevation_seq, excavator_id):
    """Run the deployed visual-only forward path used by the visualizer."""
    return model(rgb_seq, elevation_seq, None, excavator_id)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize V9, V10, or V11 YOLO-ST-VLA checkpoints")
    parser.add_argument("--checkpoint", type=str, default="output/YOLO_ST-VLA/yolo_checkpoint_best.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/excavator-motion/data/75/xcmg_data_2025-04-11-17-46-49.hdf5")
    parser.add_argument("--out_dir", type=str, default="output/yolo_vis")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_masks", action="store_true", help="Skip mask overlay")
    parser.add_argument("--mask_view", type=str, default=None,
                        choices=["last", "avg"],
                        help="V10/V11 default: last (raw); V9 default: avg")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load checkpoint ──
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    sd_keys = set(state_dict.keys())

    # ── Detect version ──
    is_v2 = any("delta_head" in k for k in sd_keys)
    is_v3 = any("action_head" in k and "action_heads" not in k for k in sd_keys)
    is_v5 = any("action_heads" in k for k in sd_keys) and not any("joint_embed" in k for k in sd_keys)
    is_v8 = any("joint_embed" in k for k in sd_keys)
    model_version = detect_model_version(sd_keys, ckpt)
    is_v11 = model_version == "v11"
    is_v10 = model_version == "v10"
    is_temporal_version = model_version in ("v10", "v11")
    v = ("V11" if is_v11 else "V10" if is_v10 else "V2" if is_v2 else
         "V3/V4" if is_v3 else "V5-V7" if not is_v8 else "V8/V9")
    print(f"  Detected: {v}")
    # V10/V11 default: raw last-frame masks; V9 default: temporal average.
    if args.mask_view is None:
        args.mask_view = "last" if is_temporal_version else "avg"

    # ── Remap old keys ──
    remapped = {}
    for k, val in list(state_dict.items()):
        if "delta_head" in k or ("action_head." in k and "action_heads" not in k):
            continue  # too old
        if "qpos_mod" in k or "qpos_proj" in k:
            continue  # V8 removed qpos modulation
        if is_v5 and "action_heads." in k:
            parts = k.split(".")
            eid = int(parts[1])
            rest = ".".join(parts[2:])
            if rest.startswith("6.") or rest.startswith("3."):
                continue  # output layer dim mismatch
            for j in range(4):
                remapped[f"action_heads.{eid}.{j}.{rest}"] = val.clone()
        if "mask_head" in k and "mask_heads" not in k:
            for j in range(4):
                remapped[k.replace("mask_head", f"mask_heads.{j}")] = val.clone()
    state_dict.update(remapped)
    if "query_tokens" in state_dict and "joint_queries" not in state_dict:
        state_dict["joint_queries"] = state_dict.pop("query_tokens")

    # ── Auto-detect config ──
    hidden_dim = state_dict.get("encoder.layers.0.self_attn.in_proj_weight",
        state_dict.get("encoder.layers.0.linear1.weight")).shape[1]
    n_layers = 4
    for k in sd_keys:
        if k.startswith("encoder.layers."):
            idx = int(k.split(".")[2])
            n_layers = max(n_layers, idx + 1)
    ff_dim = state_dict.get("encoder.layers.0.linear1.weight").shape[0]
    print(f"  Config: hidden_dim={hidden_dim}, n_layers={n_layers}, ff_dim={ff_dim}")

    G = args.img_size // 16

    model = ExcavatorVLAYolo(
        seq_len=args.seq_len, img_size=args.img_size,
        hidden_dim=hidden_dim, n_heads=8,
        n_layers=n_layers, ff_dim=ff_dim,
        dropout=0.0, pretrained=False,
        version=model_version if is_temporal_version else "v9",
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Model loaded. Grid={G}×{G}")

    # ── Load data ──
    print(f"Loading: {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        # Target = next frame's absolute joint angle (matching dataset.py)
        targets = np.zeros_like(qpos)
        targets[:-1] = qpos[1:]
        targets[-1] = qpos[-1]

    N = len(targets)
    T = args.seq_len
    print(f"  {N} frames")

    # ── Parse excavator ID ──
    path_lower = args.data_path.lower()
    if '/75/' in path_lower or '\\75\\' in path_lower:
        excv_id = 0
    elif '/306/' in path_lower or '\\306\\' in path_lower:
        excv_id = 1
    elif '/490/' in path_lower or '\\490\\' in path_lower:
        excv_id = 2
    else:
        excv_id = 3
    excv_tensor = torch.tensor([excv_id], dtype=torch.long).to(device)

    # ── Preprocess images ──
    print("Preprocessing images...")
    mem_needed = N * 3 * args.img_size * args.img_size * 4 / (1024**2)
    if mem_needed > 2000:
        print(f"  Large ({mem_needed:.0f}MB), on-the-fly mode")
        rgb_pp = None
    else:
        rgb_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
        elev_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
        for i in tqdm(range(N), desc="  Preprocessing"):
            rgb_pp[i] = preprocess_image(mains[i], args.img_size)
            elev_pp[i] = preprocess_image(elevations[i], args.img_size)

    # ── Inference ──
    print("Running inference...")
    predictions = np.full((N, 4), np.nan, dtype=np.float32)
    all_masks = np.zeros((N, 4, G, G), dtype=np.float32)
    predictions[:T - 1] = targets[:T - 1]

    for start in tqdm(range(0, N - T), desc="  Inference"):
        end = start + T
        if rgb_pp is not None:
            rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
            elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        else:
            _rgb = np.zeros((T, 3, args.img_size, args.img_size), dtype=np.float32)
            _elev = np.zeros((T, 3, args.img_size, args.img_size), dtype=np.float32)
            for t in range(T):
                _rgb[t] = preprocess_image(mains[start + t], args.img_size)
                _elev[t] = preprocess_image(elevations[start + t], args.img_size)
            rgb_seq = torch.from_numpy(_rgb).unsqueeze(0).to(device)
            elev_seq = torch.from_numpy(_elev).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = run_visual_inference(model, rgb_seq, elev_seq, excv_tensor)
            raw_out = outputs[0]
            masks_spatial_raw = outputs[2]                                                        # [1,4,T,G,G]
            masks_data = select_mask_view(masks_spatial_raw, args.mask_view)  # [1,4,G,G]

        tgt_idx = start + T - 1
        # V2: decode_delta, V3+: decode_action
        if hasattr(model, 'decode_action'):
            action_pred = model.decode_action(raw_out)[0].cpu().numpy()
        else:
            action_pred = model.decode_delta(raw_out)[0].cpu().numpy()
        predictions[tgt_idx] = action_pred
        all_masks[tgt_idx] = masks_data[0].cpu().numpy()  # [4, G, G]

    # ── MAE ──
    valid = ~np.isnan(predictions[:, 0])
    err = np.abs(predictions[valid] - targets[valid])
    mae = err.mean(axis=0)
    print(f"\nMAE: Boom={mae[0]:.4f} Arm={mae[1]:.4f} Bucket={mae[2]:.4f} Swing={mae[3]:.4f}")
    print(f"  Mean: {mae.mean():.4f} rad = {mae.mean()*57.3:.2f}°")

    # ── Render video ──
    print("Rendering video...")
    timeline = np.arange(N, dtype=np.float32)
    curve_h = CURVE_H_PER_JOINT * 4

    if not args.no_masks:
        mask_w = MASK_W
        mask_h = MASK_H
        img_row_w = MAIN_W + 2 * mask_w
        top_h = MAIN_H + ELEV_H + PAD
    else:
        img_row_w = MAIN_W + ELEV_W
        mask_w = 0
        mask_h = 0
        top_h = MAIN_H

    title_h = 30
    total_w = max(img_row_w, CURVE_W)
    frames = []

    for i in tqdm(range(T - 1, N), desc="  Rendering"):
        main_rgb = cv2.cvtColor(mains[i], cv2.COLOR_BGR2RGB)
        main_rgb = resize_keep_aspect(main_rgb, MAIN_W, MAIN_H)
        elev = cv2.cvtColor(elevations[i], cv2.COLOR_BGR2RGB)
        elev = resize_keep_aspect(elev, ELEV_W, ELEV_H)

        if not args.no_masks:
            # Render 4 mask panels
            masks_i = all_masks[i]  # [4, G, G]
            mask_panels = []

            raw_display = cv2.resize(mains[i], (mask_w, mask_h))
            for k in range(4):
                panel = render_masks(raw_display, masks_i[k], k, G)
                # Add region label
                cv2.putText(panel, REGION_NAMES[k], (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                mask_panels.append(panel)

            top_row = np.concatenate([main_rgb] + mask_panels[:2], axis=1)
            bot_row = np.concatenate([elev, mask_panels[2], mask_panels[3]], axis=1)
            top_section = np.concatenate([top_row, bot_row], axis=0)
        else:
            top_section = np.concatenate([main_rgb, elev], axis=1)

        top_section_w = top_section.shape[1]
        section_padded = cv2.copyMakeBorder(
            top_section, 0, 0, 0, max(0, total_w - top_section_w),
            cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Title
        title_img = np.full((title_h, total_w, 3), 255, dtype=np.uint8)
        cv2.putText(title_img, f"Frame: {i} / {N}  |  MAE: Boom={mae[0]:.4f} Arm={mae[1]:.4f} "
                    f"Bucket={mae[2]:.4f} Swing={mae[3]:.4f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        # Curves
        curve_img = render_curves(timeline, targets, predictions, i, frame_range=200)
        ch, cw = curve_img.shape[:2]
        if cw != total_w:
            curve_img = cv2.resize(curve_img, (total_w, ch))

        frame = np.concatenate([title_img, section_padded,
                                np.full((PAD, total_w, 3), 255, dtype=np.uint8),
                                curve_img], axis=0)
        frames.append(frame)

    # Save
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(args.data_path).stem
    out_path = f"{args.out_dir}/{name}_yolo_gt_vs_pred.mp4"
    imageio.mimsave(out_path, frames, fps=args.fps, macro_block_size=1)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
