"""SCI-style prediction-curve plot from pre-computed inference CSV + meta.

Usage:
  python vla_model/plot_curves.py \
    --predictions_csv export/episode_predictions.csv \
    --meta_json export/episode_meta.json \
    --out figure/curves_75.png
"""

import sys, argparse, json, numpy as np, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.right': False, 'axes.spines.top': False,
})

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
JOINT_KEYS  = ['boom', 'arm', 'bucket', 'swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
GT_COLOR = '#333333'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_csv', required=True)
    parser.add_argument('--meta_json', required=True)
    parser.add_argument('--out', default='figure/curves.png')
    parser.add_argument('--frames_to_plot', type=int, default=300)
    args = parser.parse_args()

    # ── Load meta ──
    with open(args.meta_json) as f:
        meta = json.load(f)
    n_frames = meta['n_frames']
    print(f"Loaded meta: {meta['excavator']} {meta['version']} {n_frames}f")

    # ── Load predictions ──
    targets = np.full((n_frames, 4), np.nan, dtype=np.float32)
    predictions = np.full((n_frames, 4), np.nan, dtype=np.float32)
    with open(args.predictions_csv) as f:
        for r in csv.DictReader(f):
            i = int(r['frame'])
            for j in range(4):
                jk = JOINT_KEYS[j]
                targets[i, j] = float(r[f'target_{jk}'])
                v = r[f'pred_{jk}']
                predictions[i, j] = float(v) if v != 'nan' else np.nan
    valid_count = np.sum(~np.isnan(predictions[:, 0]))
    print(f'Loaded predictions: {n_frames} frames, {valid_count} predicted')

    # ── Build figure ──
    N = n_frames
    T = 8
    n_curve = min(args.frames_to_plot, N) if args.frames_to_plot > 0 else N
    timeline = np.arange(n_curve, dtype=np.float32)
    sf = T - 1
    ef = min(N, sf + n_curve)

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    for j, ax in enumerate(axes):
        ax.set_facecolor('white')

        ax.plot(timeline, targets[sf:ef, j],
                color=GT_COLOR, linewidth=0.8, alpha=0.9, label='Ground Truth')
        ax.plot(timeline, predictions[sf:ef, j],
                color=JOINT_COLORS[j], linewidth=1.0, alpha=0.85, label='Prediction')

        vj = ~np.isnan(predictions[sf:ef, j])
        if vj.any():
            ax.fill_between(timeline[vj],
                            targets[sf:ef, j][vj],
                            predictions[sf:ef, j][vj],
                            alpha=0.08, color=JOINT_COLORS[j])

        ax.text(0.99, 0.92, JOINT_NAMES[j],
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.12, color='#999')
        ax.tick_params(labelsize=9)

        if j == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=False,
                      bbox_to_anchor=(0, 1.25))

    axes[-1].set_xlabel('Frame', fontsize=13, fontweight='bold')
    axes[-1].set_xlim(0, len(timeline) - 1)

    fig.suptitle(f"Per-joint Prediction  —  Excavator {meta['excavator']}",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.8, h_pad=0.3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
