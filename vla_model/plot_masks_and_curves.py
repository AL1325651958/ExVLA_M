"""SCI-style mask + prediction-curve composite figure from visualization frames.

Reads a checkpoint, runs inference on one episode, renders masks and GT-vs-Pred curves.
Outputs a single publication-ready 2×1 layout: masks (top) + 4-DOF curves (bottom).
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
import matplotlib.ticker as mticker

# ── SCI style ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
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


def preprocess_image(img_bgr, size=224):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


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


def render_prediction_curves(timeline, targets, predictions, n_frames=300):
    """Render 4-DOF GT vs Prediction curves in SCI style."""
    N = len(timeline)
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    for j in range(4):
        ax = axes[j]
        ax.set_facecolor('white')
        ax.plot(timeline, targets[:, j], color=GT_COLOR, linewidth=0.8, alpha=0.9, label='Ground Truth')
        ax.plot(timeline, predictions[:, j], color=JOINT_COLORS[j], linewidth=0.8, alpha=0.8, label='Prediction')

        # shade error
        valid = ~np.isnan(predictions[:, j])
        if valid.any():
            ax.fill_between(timeline[valid], targets[valid, j], predictions[valid, j],
                           alpha=0.12, color=JOINT_COLORS[j])

        ax.set_ylabel(JOINT_NAMES[j], fontsize=9, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.15, color='#999')
        ax.tick_params(labelsize=7)

        # Compute MAE
        err = np.abs(predictions[valid, j] - targets[valid, j]).mean()
        ax.text(0.99, 0.05, f'MAE={err:.4f} rad', transform=ax.transAxes, ha='right',
               fontsize=7, color='#555', style='italic')

        if j == 0:
            ax.legend(loc='upper right', fontsize=7, frameon=False)

    axes[-1].set_xlabel('Frame', fontsize=9, fontweight='bold')
    axes[-1].set_xlim(timeline[0], timeline[-1])
    fig.tight_layout(pad=0.5, h_pad=0.2)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out', default='figure/mask_curves.png')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frames_to_plot', type=int, default=300,
                       help='How many frames to show on curves (0=all)')
    parser.add_argument('--mask_frames', type=str, default='0,3,6',
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
    print(f'  {version_tag} | n_layers={n_layers} | hidden_dim={hidden_dim}')

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

    # ── Build figure ──
    print('Building figure...')
    mask_frame_indices = [int(s) for s in args.mask_frames.split(',') if s.strip()]
    mask_frame_indices = [i for i in mask_frame_indices if T - 1 <= i < N]

    n_mask_frames = len(mask_frame_indices)
    n_rows = 1 + n_mask_frames  # curves + mask rows
    fig = plt.figure(figsize=(14, 3.5 * n_rows))

    # ── Row 1: Prediction curves ──
    n_curve_frames = min(args.frames_to_plot, N) if args.frames_to_plot > 0 else N
    timeline = np.arange(n_curve_frames, dtype=np.float32)
    sf = max(0, T - 1)
    ef = min(N, sf + n_curve_frames)

    gs = fig.add_gridspec(n_rows, 1, hspace=0.35)

    # Curves subplot
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_curves = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.15)
    curve_axes = [fig.add_subplot(gs_curves[j]) for j in range(4)]

    pred_slice = predictions[sf:ef]
    tgt_slice = targets[sf:ef]
    for j, ax in enumerate(curve_axes):
        ax.set_facecolor('white')
        ax.plot(timeline, tgt_slice[:, j], color=GT_COLOR, linewidth=0.6, alpha=0.85, label='GT')
        ax.plot(timeline, pred_slice[:, j], color=JOINT_COLORS[j], linewidth=0.7, alpha=0.8, label='Pred')
        valid = ~np.isnan(pred_slice[:, j])
        if valid.any():
            ax.fill_between(timeline[valid], tgt_slice[valid, j], pred_slice[valid, j],
                           alpha=0.10, color=JOINT_COLORS[j])
        err = np.abs(pred_slice[valid, j] - tgt_slice[valid, j]).mean()
        ax.text(0.99, 0.95, f'{JOINT_NAMES[j]}  MAE={err:.4f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=7, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.12, color='#999')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(loc='upper left', fontsize=6, frameon=False, bbox_to_anchor=(0, 1.3))
        if j < 3: ax.set_xticklabels([])
    curve_axes[-1].set_xlabel('Frame', fontsize=9, fontweight='bold')
    curve_axes[-1].set_xlim(0, len(timeline) - 1)

    # ── Rows 2+: Mask overlay ──
    row_idx = 1
    for fidx in mask_frame_indices:
        # Show original RGB + Elev + 4 joint mask pairs
        gs_masks = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[row_idx], wspace=0.03)

        # Original images
        rgb_orig = cv2.cvtColor(mains[fidx], cv2.COLOR_BGR2RGB)
        elev_orig = cv2.cvtColor(elevations[fidx], cv2.COLOR_BGR2RGB)
        ax_rgb = fig.add_subplot(gs_masks[0]); ax_rgb.imshow(rgb_orig); ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        ax_rgb.set_title('RGB', fontsize=7, fontweight='bold')
        ax_elv = fig.add_subplot(gs_masks[1]); ax_elv.imshow(elev_orig); ax_elv.set_xticks([]); ax_elv.set_yticks([])
        ax_elv.set_title('Elevation', fontsize=7, fontweight='bold')

        # 4 joint mask overlays (RGB overlay with mask)
        for j in range(4):
            ax_m = fig.add_subplot(gs_masks[j + 2])
            mask_14 = all_masks[fidx, 0, j] if is_v16 else all_masks[fidx, 0, j]
            overlayed = render_mask_overlay(rgb_orig, mask_14,
                                           [int(c) for c in JOINT_COLORS[j].lstrip('#')])
            # Convert BGR→RGB
            overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB) if overlayed.shape[-1] == 3 else overlayed
            ax_m.imshow(overlayed_rgb)
            ax_m.set_xticks([]); ax_m.set_yticks([])
            ax_m.set_title(JOINT_NAMES[j], fontsize=7, fontweight='bold',
                          color=JOINT_COLORS[j])

        row_idx += 1

    # Title
    ep_name = Path(args.data_path).stem
    excv_label = {0: '75', 1: '306', 2: '490'}.get(excv_id, '?')
    fig.suptitle(f'{version_tag} — Excavator {excv_label} — {ep_name}',
                fontsize=11, fontweight='bold', y=1.02)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
