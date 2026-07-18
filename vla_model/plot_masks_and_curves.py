"""Plot mask & curve figure from pre-computed inference CSVs + NPY + meta.

Usage after running export_inference.py:
  python vla_model/plot_masks_and_curves.py \
    --predictions_csv export/episode_predictions.csv \
    --masks_npy export/episode_masks.npy \
    --meta_json export/episode_meta.json \
    --data_path data/excavator-motion/data/75/file.hdf5 \
    --out figure/mask_curves.png
"""

import sys, argparse, json, h5py, numpy as np, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

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
JOINT_KEYS = ['boom', 'arm', 'bucket', 'swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
GT_COLOR = '#333333'


def _hex_to_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def render_mask_overlay(bgr_img, mask_14x14, color_bgr, alpha=0.70):
    h, w = bgr_img.shape[:2]
    mask = cv2.resize(mask_14x14, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0, 1)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    overlay = bgr_img.astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha * mask) + color_bgr[c] * (alpha * mask)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_csv', required=True, help='export_inference.py output CSV')
    parser.add_argument('--masks_npy', required=True, help='export_inference.py output .npy')
    parser.add_argument('--meta_json', required=True, help='export_inference.py output meta')
    parser.add_argument('--data_path', required=True, help='original .hdf5 episode (for original images)')
    parser.add_argument('--out', default='figure/mask_curves.png')
    parser.add_argument('--frames_to_plot', type=int, default=300)
    parser.add_argument('--mask_frames', type=str, default='100,200,300')
    args = parser.parse_args()

    # ── Load meta ──
    with open(args.meta_json) as f:
        meta = json.load(f)
    n_frames = meta['n_frames']
    r2_per_joint = [meta['r2'][k] for k in JOINT_KEYS]
    print(f"Loaded meta: {meta['excavator']} {meta['version']} {n_frames}f  "
          f"R²={[f'{r:.4f}' for r in r2_per_joint]}")

    # ── Load predictions ──
    targets = np.full((n_frames, 4), np.nan, dtype=np.float32)
    predictions = np.full((n_frames, 4), np.nan, dtype=np.float32)
    import csv
    with open(args.predictions_csv) as f:
        for r in csv.DictReader(f):
            i = int(r['frame'])
            for j in range(4):
                jk = JOINT_KEYS[j]
                targets[i, j] = float(r[f'target_{jk}'])
                v = r[f'pred_{jk}']
                predictions[i, j] = float(v) if v != 'nan' else np.nan
    print(f'Loaded predictions: {n_frames} frames, {np.sum(~np.isnan(predictions[:,0]))} predicted')

    # ── Load masks ──
    all_masks = np.load(args.masks_npy)   # [N, 2, 4, 14, 14]
    print(f'Loaded masks: {all_masks.shape}')

    # ── Load original images ──
    print(f'Loading original images: {args.data_path}')
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
    N = len(mains)
    T = 8

    # ── Parse mask frames ──
    mask_frames = [int(s) for s in args.mask_frames.split(',') if s.strip()]
    mask_frames = [i for i in mask_frames if T - 1 <= i < N]
    n_mask_rows = len(mask_frames)

    # ── Build figure ──
    fig_w = 8.5
    curve_h = 4.0                    # taller to accommodate inter-joint spacing
    mask_row_h = 1.2
    title_h = 0.3
    fig_h = title_h + curve_h + n_mask_rows * mask_row_h + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1 + n_mask_rows, 1, hspace=0.28,
                          height_ratios=[curve_h] + [mask_row_h] * n_mask_rows)

    # ── Row 1: Prediction curves ──
    n_curve = min(args.frames_to_plot, N) if args.frames_to_plot > 0 else N
    timeline = np.arange(n_curve, dtype=np.float32)
    sf = T - 1; ef = min(N, sf + n_curve)

    gs_curves = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.35)
    curve_axes = [fig.add_subplot(gs_curves[j]) for j in range(4)]

    for j, ax in enumerate(curve_axes):
        ax.set_facecolor('white')
        ax.plot(timeline, targets[sf:ef, j], color=GT_COLOR, linewidth=0.5, alpha=0.85, label='GT')
        ax.plot(timeline, predictions[sf:ef, j], color=JOINT_COLORS[j], linewidth=0.65, alpha=0.82, label='Pred')
        vj = ~np.isnan(predictions[sf:ef, j])
        if vj.any():
            ax.fill_between(timeline[vj], targets[sf:ef, j][vj], predictions[sf:ef, j][vj],
                           alpha=0.10, color=JOINT_COLORS[j])
        ax.text(0.99, 0.94, f'{JOINT_NAMES[j]}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.12, color='#999')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(loc='upper left', fontsize=6, frameon=False, bbox_to_anchor=(0, 1.35))
        if j < 3:
            ax.set_xticklabels([])
    curve_axes[-1].set_xlabel('Frame', fontsize=9, fontweight='bold')
    curve_axes[-1].set_xlim(0, len(timeline) - 1)

    # ── Rows 2+: Mask overlay ──
    for ri, fidx in enumerate(mask_frames):
        gs_masks = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[ri + 1], wspace=0.04)

        rgb_orig = cv2.cvtColor(mains[fidx], cv2.COLOR_BGR2RGB)
        elev_orig = cv2.resize(cv2.cvtColor(elevations[fidx], cv2.COLOR_BGR2RGB),
                               (rgb_orig.shape[1], rgb_orig.shape[0]))

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

    # Close HDF5 (opened for elevation)
    fig.suptitle(f"Per-joint Prediction  —  Excavator {meta['excavator']}",
                 fontsize=11, fontweight='bold', y=0.995)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
