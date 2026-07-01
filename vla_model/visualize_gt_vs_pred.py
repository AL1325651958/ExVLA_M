"""Generate video: GT vs Pred joint curves alongside RGB + elevation views."""

import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import cv2
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model import ExcavatorVLA
from vla_model.config import Config
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
GT_COLOR = '#333333'
PRED_COLOR = '#e74c3c'
# Step 0 (closest) → Step N-1 (furthest): dark → light red
PRED_STEP_ALPHAS = [1.0, 0.70, 0.50, 0.35, 0.25, 0.18, 0.13, 0.09, 0.06, 0.04]
PRED_STEP_WIDTHS = [1.2, 0.70, 0.55, 0.45, 0.35, 0.30, 0.25, 0.22, 0.19, 0.16]

MAIN_H, MAIN_W = 270, 360
ELEV_H, ELEV_W = 270, 360
CURVE_W = 740
CURVE_H_PER_JOINT = 110
PAD = 8


def preprocess_image(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def resize_keep_aspect(img, target_w, target_h):
    """Resize image to fit target size with padding."""
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


def render_curves(timeline, targets, predictions, current_idx, frame_range=200):
    """Render 4 joint curves with GT (grey) and multi-step Pred (red gradient).

    Args:
        predictions: [N, K, 4]  — K prediction steps per frame (K=action_chunk)
    """
    N = len(timeline)
    K = predictions.shape[1]  # number of prediction steps
    half = frame_range // 2
    start = max(0, current_idx - half)
    end   = min(N, current_idx + half)
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

        # Full GT (light grey background reference)
        ax.plot(t_slice, targets[start:end, j], color='#cccccc', linewidth=0.6, alpha=0.7)

        # GT up to now (dark grey solid)
        ax.plot(t_past[start:], gt_past[start:], color=GT_COLOR, linewidth=1.0, label='GT')

        # Prediction: K steps, decreasing opacity
        for k in range(K):
            pred_step = predictions[:current_idx + 1, k, j]
            alpha = PRED_STEP_ALPHAS[min(k, len(PRED_STEP_ALPHAS) - 1)]
            lw    = PRED_STEP_WIDTHS[min(k, len(PRED_STEP_WIDTHS) - 1)]
            label = 'Pred(step1)' if k == 0 else None
            linestyle = '-' if k == 0 else '--'
            ax.plot(t_past[start:], pred_step[start:], color=PRED_COLOR,
                    linewidth=lw, alpha=alpha, linestyle=linestyle, label=label)

        # Current frame markers
        ax.axvline(x=timeline[current_idx], color='#3498db', linewidth=1.5, linestyle='--', alpha=0.8)

        # GT dot at current frame
        ax.plot(timeline[current_idx], targets[current_idx, j], 'o', color=GT_COLOR, markersize=5)

        # Prediction dots: step 0 big, rest small (skip NaN = not yet available)
        val0 = predictions[current_idx, 0, j]
        if not np.isnan(val0):
            ax.plot(timeline[current_idx], val0, 'o', color=PRED_COLOR, markersize=5)
        for k in range(1, K):
            val = predictions[current_idx, k, j]
            if not np.isnan(val):
                alpha = PRED_STEP_ALPHAS[min(k, len(PRED_STEP_ALPHAS) - 1)]
                ax.plot(timeline[current_idx], val, 'o', color=PRED_COLOR, markersize=3, alpha=alpha)

        ax.set_xlim(x_min, x_max)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=8, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.2, color='#999')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')

        if j == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/checkpoints/checkpoint_best.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/excavator-motion/data/75/xcmg_data_2025-04-11-17-46-49.hdf5")
    parser.add_argument("--out_dir", type=str, default="output/excavator_vis")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pred_steps", type=int, default=None,
                        help="Number of prediction steps to draw (default: use model's action_chunk)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config()
    if "config" in ckpt and hasattr(ckpt["config"], "hidden_dim"):
        config = ckpt["config"]

    # Detect model version from state_dict keys
    state_keys = set(ckpt["model_state_dict"].keys())
    is_delta_model = any("delta_head" in k for k in state_keys)
    is_v2_model = any("action_head" in k for k in state_keys)
    if is_delta_model:
        action_chunk = 1
        print("Model type: delta prediction (v3)")
    elif is_v2_model:
        action_chunk = config.action_chunk if hasattr(config, "action_chunk") else 1
        print(f"Model type: action chunking (v2), action_chunk={action_chunk}")
    else:
        # Fallback: treat as delta if model doesn't accept action_chunk
        action_chunk = 1
        print("Model type: unknown, treating as delta")

    model = ExcavatorVLA(
        seq_len=args.seq_len,
        hidden_dim=config.hidden_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ff_dim=config.ff_dim,
        dropout=0.0, pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # For delta model, pred_steps is always 1
    if is_delta_model:
        pred_steps = 1
    else:
        pred_steps = min(args.pred_steps or action_chunk, action_chunk)
    print(f"Prediction steps to visualize: {pred_steps}")

    # Load data
    print(f"Loading data: {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        if 'action' in f:
            targets = f['action'][:].astype(np.float32)
        else:
            # 306 format: no action key, use next qpos as target
            targets = np.zeros_like(qpos)
            targets[:-1] = qpos[1:]
            targets[-1] = qpos[-1]

    # Parse excavator ID from path
    path_lower = args.data_path.lower()
    if '/75/' in path_lower or '\\75\\' in path_lower:
        excv_id = 0
    elif '/306/' in path_lower or '\\306\\' in path_lower:
        excv_id = 1
    elif '/490/' in path_lower or '\\490\\' in path_lower:
        excv_id = 2
    else:
        excv_id = 3
    N = len(targets)

    # Preprocess images — on-the-fly if >2GB
    print("Preprocessing images...")
    T_img = args.seq_len
    qpos_pp = qpos.astype(np.float32)
    mem_needed = N * 3 * args.img_size * args.img_size * 4 / (1024**2)
    if mem_needed > 2000:
        print(f"  Large ({mem_needed:.0f}MB), on-the-fly mode")
        rgb_pp = None; elev_pp = None
    else:
        rgb_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
        elev_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
        for i in range(N):
            rgb_pp[i] = preprocess_image(mains[i], args.img_size)
            elev_pp[i] = preprocess_image(elevations[i], args.img_size)

    excv_tensor = torch.tensor([excv_id], dtype=torch.long).to(device)

    # Run sliding-window inference
    print("Running inference...")
    predictions = np.full((N, pred_steps, 4), np.nan, dtype=np.float32)

    for start in range(0, N - T_img):
        end = start + T_img
        if rgb_pp is not None:
            rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
            elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        else:
            _rgb = np.zeros((T_img, 3, args.img_size, args.img_size), dtype=np.float32)
            _elev = np.zeros((T_img, 3, args.img_size, args.img_size), dtype=np.float32)
            for t in range(T_img):
                i = start + t
                _rgb[t] = preprocess_image(mains[i], args.img_size)
                _elev[t] = preprocess_image(elevations[i], args.img_size)
            rgb_seq = torch.from_numpy(_rgb).unsqueeze(0).to(device)
            elev_seq = torch.from_numpy(_elev).unsqueeze(0).to(device)
        qpos_seq = torch.from_numpy(qpos_pp[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(rgb_seq, elev_seq, qpos_seq, excv_tensor)  # [1, K, 4] or [1, 4]

        tgt_idx = start + T_img - 1

        if is_delta_model:
            # Model predicts delta (change from last qpos)
            # Convert to absolute: last_qpos + delta
            delta = pred[0].cpu().numpy()  # [4]
            last_qpos = qpos_pp[tgt_idx]     # [4]
            absolute_pred = last_qpos + delta
            predictions[tgt_idx, 0] = absolute_pred
        else:
            for k in range(pred_steps):
                tk = tgt_idx + k
                if tk < N:
                    predictions[tk, k] = pred[0, k].cpu().numpy()

    # Per-step MAE
    print(f"Per-step MAE ({pred_steps} steps):")
    mae_per_step = []
    for k in range(pred_steps):
        mask = ~np.isnan(predictions[:, k, 0])
        if mask.sum() > 0:
            err_k = np.abs(predictions[mask, k] - targets[mask])
            mae_k = err_k.mean(axis=0)
            mae_per_step.append(mae_k)
            print(f"  Step {k+1}: Boom={mae_k[0]:.4f} Arm={mae_k[1]:.4f} "
                  f"Bucket={mae_k[2]:.4f} Swing={mae_k[3]:.4f}  (n={mask.sum()})")

    # Overall MAE (step 0 only)
    mae_per = mae_per_step[0] if mae_per_step else np.zeros(4)
    print(f"Overall (step1 only): Boom={mae_per[0]:.4f} Arm={mae_per[1]:.4f} "
          f"Bucket={mae_per[2]:.4f} Swing={mae_per[3]:.4f} rad")

    # ============ Render video ============
    print("Rendering video frames...")
    timeline = np.arange(N, dtype=np.float32)
    curve_h = CURVE_H_PER_JOINT * 4
    img_w = MAIN_W + ELEV_W  # 720
    total_h = MAIN_H + PAD + curve_h + PAD

    # Title bar
    title_h = 30
    total_h += title_h

    frames = []

    for i in range(T_img - 1, N):
        # --- Top: current RGB + current Elevation ---
        main_rgb = mains[i].copy()  # BGR
        main_rgb = cv2.cvtColor(main_rgb, cv2.COLOR_BGR2RGB)
        main_rgb = resize_keep_aspect(main_rgb, MAIN_W, MAIN_H)

        elev = elevations[i].copy()
        elev = cv2.cvtColor(elev, cv2.COLOR_BGR2RGB)
        elev = resize_keep_aspect(elev, ELEV_W, ELEV_H)

        top_row = np.concatenate([main_rgb, elev], axis=1)  # [H, 720, 3]
        total_w = top_row.shape[1]

        # Title
        title_img = np.full((title_h, total_w, 3), 255, dtype=np.uint8)
        cv2.putText(title_img, f"Frame: {i} / {N}  |  Model: {'delta' if is_delta_model else 'chunk'}  |  GT vs Prediction",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

        # Step1 MAE text (comparable to V1 single-step)
        mae_text = (f"Step1 MAE: Boom={mae_per[0]:.4f}  Arm={mae_per[1]:.4f}  "
                    f"Bucket={mae_per[2]:.4f}  Swing={mae_per[3]:.4f}")
        text_w = cv2.getTextSize(mae_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(title_img, mae_text, (total_w - text_w - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # --- Bottom: Joint curves ---
        curve_img = render_curves(timeline, targets, predictions, i, frame_range=200)

        # Pad/resize curve to match total_w
        ch, cw = curve_img.shape[:2]
        if cw != total_w:
            curve_img = cv2.resize(curve_img, (total_w, ch))

        # Assemble
        frame = np.concatenate([title_img, top_row,
                                np.full((PAD, total_w, 3), 255, dtype=np.uint8),
                                curve_img], axis=0)

        frames.append(frame)

        if (i - T_img + 2) % 100 == 0:
            print(f"  Frame {i}/{N}")

    # Save video
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(args.data_path).stem
    out_path = f"{args.out_dir}/{name}_gt_vs_pred.mp4"
    imageio.mimsave(out_path, frames, fps=args.fps, macro_block_size=1)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
