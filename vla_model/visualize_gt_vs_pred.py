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
    """Render 4 joint curves with GT (grey) and Pred (red)."""
    N = len(timeline)
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

        # GT curve (grey, up to current)
        t_past = timeline[:current_idx + 1]
        gt_past = targets[:current_idx + 1, j]
        pred_past = predictions[:current_idx + 1, j]

        # Full GT (light grey background reference)
        ax.plot(t_slice, targets[start:end, j], color='#cccccc', linewidth=0.6, alpha=0.7)

        # GT up to now (dark grey)
        ax.plot(t_past[start:], gt_past[start:], color=GT_COLOR, linewidth=1.0, label='GT')

        # Pred up to now (red)
        ax.plot(t_past[start:], pred_past[start:], color=PRED_COLOR, linewidth=1.0, label='Pred')

        # Current frame marker
        ax.axvline(x=timeline[current_idx], color='#3498db', linewidth=1.5, linestyle='--', alpha=0.8)

        # Current value dots
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
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config()
    if "config" in ckpt and hasattr(ckpt["config"], "hidden_dim"):
        config = ckpt["config"]

    model = ExcavatorVLA(
        seq_len=args.seq_len, action_chunk=config.action_chunk,
        hidden_dim=config.hidden_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ff_dim=config.ff_dim,
        dropout=0.0, pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
            targets[:-1] = qpos[1:]
            targets[-1] = qpos[-1]
    N = len(targets)

    # Preprocess images
    print("Preprocessing images...")
    T_img = args.seq_len
    qpos_pp = qpos.astype(np.float32)
    rgb_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
    elev_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
    for i in range(N):
        rgb_pp[i] = preprocess_image(mains[i], args.img_size)
        elev_pp[i] = preprocess_image(elevations[i], args.img_size)

    excv_tensor = torch.tensor([excv_id], dtype=torch.long).to(device)

    # Run sliding-window inference
    print("Running inference...")
    predictions = np.zeros((N, 4), dtype=np.float32)
    predictions[:T_img - 1] = targets[:T_img - 1]

    for start in range(0, N - T_img):
        end = start + T_img
        rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
        elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        qpos_seq = torch.from_numpy(qpos_pp[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(rgb_seq, elev_seq, qpos_seq, excv_tensor)
        predictions[start + T_img - 1] = pred[0, 0].cpu().numpy()

    # Error
    abs_error = np.abs(predictions - targets)
    mae_per = abs_error.mean(axis=0)
    print(f"MAE: Boom={mae_per[0]:.4f} Arm={mae_per[1]:.4f} Bucket={mae_per[2]:.4f} Swing={mae_per[3]:.4f} rad")

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
        cv2.putText(title_img, f"Frame: {i} / {N}  |  GT vs Prediction",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

        # MAE text
        mae_text = f"MAE: Boom={mae_per[0]:.4f}  Arm={mae_per[1]:.4f}  Bucket={mae_per[2]:.4f}  Swing={mae_per[3]:.4f}"
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
