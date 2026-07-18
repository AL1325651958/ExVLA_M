"""SCI-style mask + prediction-curve composite figure.

Reads a checkpoint, runs inference on one episode, renders mask overlays and
per-joint GT-vs-Pred curves.  Outputs a publication-ready figure.

Layout (top → bottom):
  1. 4-row prediction curves with per-joint R² annotations
  2. N mask-overlay rows — each row: [RGB] [Elev] [Boom mask] [Arm mask] [Bucket mask] [Swing mask]
"""

import sys, argparse, h5py, numpy as np, torch, cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vla_model.model_yolo import (
    ExcavatorVLAYolo, load_compatible_state_dict,
    upgrade_legacy_v17_1_state_dict,
)
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ── SCI style ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.right': False, 'axes.spines.top': False,
})

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
GT_COLOR = '#333333'
PRED_COLOR = '#e74c3c'


def _circular_error(pred_rad, gt_rad):
    delta = pred_rad - gt_rad
    return np.arctan2(np.sin(delta), np.cos(delta))


def preprocess_image(img_bgr, size=224):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def detect_version(state_keys, checkpoint_version=None):
    if str(checkpoint_version).lower() in ("v17.3", "v17.1"):
        return "V17.1", True, "v17.1"
    has_v17 = any("joint_logit_bias" in k for k in state_keys)
    has_temp = any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys)
    if has_v17 and has_temp: return "V17.1", True, "v17.1"
    if has_v17: return "V17", True, "v17"
    if any("rgb_proj" in k or "cross_rgb_from_elev" in k for k in state_keys):
        return "V16", True, "v16"
    if any(k.startswith("motion_adapter.") for k in state_keys): return "V11", False, "v11"
    if any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys): return "V10", False, "v10"
    return "V9", False, "v9"


def infer_config(state_dict):
    attn = state_dict.get("encoder.layers.0.self_attn.in_proj_weight",
                          state_dict.get("encoder.layers.0.linear1.weight"))
    layers = {int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")}
    return attn.shape[1], max(layers) + 1 if layers else 4, \
           state_dict.get("encoder.layers.0.linear1.weight", attn).shape[0]


def render_mask_overlay(bgr_img, mask_14x14, color, alpha=0.45):
    """Overlay a 14×14 mask onto a full-resolution image."""
    h, w = bgr_img.shape[:2]
    mask = cv2.resize(mask_14x14, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0, 1)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    overlay = bgr_img.astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha * mask) + color[c] * (alpha * mask)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _compute_per_joint_r2(predictions, targets):
    """Compute per-joint R². Swing uses circular error."""
    valid = ~np.isnan(predictions[:, 0])
    pred = predictions[valid]; gt = targets[valid]; n = len(pred)
    r2 = np.zeros(4)
    for j in range(3):
        ss_res = ((pred[:, j] - gt[:, j]) ** 2).sum()
        ss_tot = max(((gt[:, j] - gt[:, j].mean()) ** 2).sum(), 1e-10)
        r2[j] = 1 - ss_res / ss_tot
    swing_err = _circular_error(pred[:, 3], gt[:, 3])
    ss_res_s = (swing_err ** 2).sum()
    s = gt[:, 3]; mean_a = np.arctan2(np.sin(s).mean(), np.cos(s).mean())
    centered = _circular_error(s, mean_a)
    ss_tot_s = max((centered ** 2).sum(), 1e-10)
    r2[3] = 1 - ss_res_s / ss_tot_s
    return r2


def _hex_to_bgr(hex_color):
    """'#e74c3c' → (B, G, R) = (0x3c, 0x4c, 0xe7)."""
    h = hex_color.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out', default='figure/mask_curves.png')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frames_to_plot', type=int, default=300,
                       help='How many frames to show on curves (0=all)')
    parser.add_argument('--mask_frames', type=str, default='50,100,150',
                       help='Comma-separated frame indices to render masks for')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load checkpoint ──
    print(f'Loading: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    version_tag, is_v16, version_arg = detect_version(set(sd.keys()), ckpt.get('model_version'))
    if version_arg in ('v17', 'v17.1'):
        sd = upgrade_legacy_v17_1_state_dict(sd)
    hidden_dim, n_layers, ff_dim = infer_config(sd)

    model = ExcavatorVLAYolo(
        seq_len=8, img_size=224, hidden_dim=hidden_dim, n_heads=8,
        n_layers=n_layers, ff_dim=ff_dim, dropout=0.0, pretrained=False,
        version=version_arg,
    ).to(device)
    load_compatible_state_dict(model, sd)
    model.eval()
    print(f'  n_layers={n_layers}  hidden_dim={hidden_dim}')

    # ── Load episode ──
    print(f'Loading: {args.data_path}')
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        targets = np.zeros_like(qpos); targets[:-1] = qpos[1:]; targets[-1] = qpos[-1]
    N, T = len(targets), 8
    print(f'  {N} frames')

    # Parse excavator
    pl = args.data_path.lower()
    if '/75/' in pl or '\\75\\' in pl: excv_id = 0
    elif '/306/' in pl or '\\306\\' in pl: excv_id = 1
    elif '/490/' in pl or '\\490\\' in pl: excv_id = 2
    else: excv_id = 3
    excv_t = torch.tensor([excv_id], dtype=torch.long).to(device)
    excv_label = {0: '75', 1: '306', 2: '490'}.get(excv_id, '?')

    # ── Preprocess ──
    print('Preprocessing...')
    rgb_pp = np.zeros((N, 3, 224, 224), dtype=np.float32)
    elev_pp = np.zeros((N, 3, 224, 224), dtype=np.float32)
    for i in tqdm(range(N), desc='  Preprocessing'):
        rgb_pp[i] = preprocess_image(mains[i])
        elev_pp[i] = preprocess_image(elevations[i])

    # ── Inference ──
    print('Inference...')
    predictions = np.full((N, 4), np.nan, dtype=np.float32)
    G = 224 // 16
    all_masks = np.zeros((N, 2, 4, G, G), dtype=np.float32)

    for start in tqdm(range(N - T), desc='  Inference'):
        end = start + T
        rgb_seq = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
        elev_seq = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            action, avg_masks, masks_spatial = model(rgb_seq, elev_seq, None, excv_t)
        predictions[start + T - 1] = model.decode_action(action)[0].cpu().numpy()
        if is_v16 or version_arg in ('v17', 'v17.1'):
            all_masks[start + T - 1] = masks_spatial[:, :, :, -1, :, :][0].cpu().numpy()
        else:
            m = masks_spatial[:, :, -1, :, :][0].cpu().numpy()
            all_masks[start + T - 1, 0] = m; all_masks[start + T - 1, 1] = m

    # ── Per-joint R² ──
    per_joint_r2 = _compute_per_joint_r2(predictions, targets)
    valid = ~np.isnan(predictions[:, 0])
    per_joint_mae = np.zeros(4)
    for j in range(3):
        per_joint_mae[j] = np.abs(predictions[valid, j] - targets[valid, j]).mean()
    per_joint_mae[3] = np.abs(_circular_error(predictions[valid, 3], targets[valid, 3])).mean()

    # ── Build figure ──
    print('Building figure...')
    mask_frame_indices = [int(s) for s in args.mask_frames.split(',') if s.strip()]
    mask_frame_indices = [i for i in mask_frame_indices if T - 1 <= i < N]
    n_mask_rows = len(mask_frame_indices)

    # SCI double-column width, per-joint layout
    fig_w = 8.5
    curve_h = 3.2                    # 4-row curve panel
    mask_row_h = 1.2                 # each mask row — compact but legible
    title_h_space = 0.4
    fig_h = title_h_space + curve_h + n_mask_rows * mask_row_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1 + n_mask_rows, 1, hspace=0.28,
                          height_ratios=[curve_h] + [mask_row_h] * n_mask_rows)

    # ── Row 1: Prediction curves with per-joint R² ──
    n_curve_frames = min(args.frames_to_plot, N) if args.frames_to_plot > 0 else N
    timeline = np.arange(n_curve_frames, dtype=np.float32)
    sf = max(0, T - 1)
    ef = min(N, sf + n_curve_frames)

    gs_curves = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.18)
    curve_axes = [fig.add_subplot(gs_curves[j]) for j in range(4)]

    pred_slice = predictions[sf:ef]
    tgt_slice = targets[sf:ef]
    for j, ax in enumerate(curve_axes):
        ax.set_facecolor('white')
        ax.plot(timeline, tgt_slice[:, j], color=GT_COLOR, linewidth=0.5, alpha=0.85, label='GT')
        ax.plot(timeline, pred_slice[:, j], color=JOINT_COLORS[j], linewidth=0.65, alpha=0.82, label='Pred')
        valid_j = ~np.isnan(pred_slice[:, j])
        if valid_j.any():
            ax.fill_between(timeline[valid_j], tgt_slice[valid_j, j], pred_slice[valid_j, j],
                           alpha=0.10, color=JOINT_COLORS[j])
        # R² annotation (per-joint, full-episode)
        ax.text(0.99, 0.94, f'{JOINT_NAMES[j]}  R²={per_joint_r2[j]:.4f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.12, color='#999')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(loc='upper left', fontsize=6, frameon=False, bbox_to_anchor=(0, 1.28))
        if j < 3: ax.set_xticklabels([])
    curve_axes[-1].set_xlabel('Frame', fontsize=9, fontweight='bold')
    curve_axes[-1].set_xlim(0, len(timeline) - 1)

    # ── Rows 2+: Mask overlay ──
    for ri, fidx in enumerate(mask_frame_indices):
        gs_masks = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[ri + 1], wspace=0.04)

        # Original images (square aspect to match SCI 1:1 panels)
        rgb_orig = cv2.cvtColor(mains[fidx], cv2.COLOR_BGR2RGB)
        elev_orig = cv2.cvtColor(elevations[fidx], cv2.COLOR_BGR2RGB)

        ax_rgb = fig.add_subplot(gs_masks[0]); ax_rgb.imshow(rgb_orig, aspect='equal')
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        ax_rgb.set_title('RGB', fontsize=7, fontweight='bold', loc='left', pad=2)

        ax_elv = fig.add_subplot(gs_masks[1]); ax_elv.imshow(elev_orig, aspect='equal')
        ax_elv.set_xticks([]); ax_elv.set_yticks([])
        ax_elv.set_title('Elevation', fontsize=7, fontweight='bold', loc='left', pad=2)

        for j in range(4):
            ax_m = fig.add_subplot(gs_masks[j + 2])
            mask_14 = all_masks[fidx, 0, j]
            overlayed = render_mask_overlay(rgb_orig, mask_14, _hex_to_bgr(JOINT_COLORS[j]))
            overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
            ax_m.imshow(overlayed_rgb, aspect='equal')
            ax_m.set_xticks([]); ax_m.set_yticks([])
            ax_m.set_title(JOINT_NAMES[j], fontsize=7, fontweight='bold',
                          color=JOINT_COLORS[j], loc='left', pad=2)

    # Title — no version, excavator label only, per-joint comparison
    ep_name = Path(args.data_path).stem
    fig.suptitle(f'逐关节预测对比  —  Excavator {excv_label}',
                 fontsize=11, fontweight='bold', y=0.995)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {args.out}')
    print(f'  Per-joint R²: {[f"{r:.4f}" for r in per_joint_r2]}')
    print(f'  Per-joint MAE: {[f"{m:.4f}" for m in per_joint_mae]}')


if __name__ == '__main__':
    main()
