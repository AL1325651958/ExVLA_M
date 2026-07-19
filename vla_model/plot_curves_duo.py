"""SCI-style side-by-side prediction-curve plot from pre-computed inference data.

Usage:
  python vla_model/plot_curves.py \
    --pred_75 export/V17_3/75/ep_predictions.csv --meta_75 export/V17_3/75/ep_meta.json \
    --pred_490 export/V17_3/490/ep_predictions.csv --meta_490 export/V17_3/490/ep_meta.json \
    --out figure/curves_duo.png
"""

import sys, argparse, json, numpy as np, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.right': False, 'axes.spines.top': False,
})

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
JOINT_KEYS  = ['boom', 'arm', 'bucket', 'swing']
JOINT_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
GT_COLOR = '#333333'


def load_data(predictions_csv, meta_json):
    with open(meta_json) as f:
        meta = json.load(f)
    n_frames = meta['n_frames']

    targets = np.full((n_frames, 4), np.nan, dtype=np.float32)
    predictions = np.full((n_frames, 4), np.nan, dtype=np.float32)
    with open(predictions_csv) as f:
        for r in csv.DictReader(f):
            i = int(r['frame'])
            for j in range(4):
                jk = JOINT_KEYS[j]
                targets[i, j] = float(r[f'target_{jk}'])
                v = r[f'pred_{jk}']
                predictions[i, j] = float(v) if v != 'nan' else np.nan
    return targets, predictions, meta


def draw_curves(axes, timeline, targets, predictions, sf, ef, r2_values):
    for j, ax in enumerate(axes):
        ax.set_facecolor('white')
        ax.plot(timeline, targets[sf:ef, j],
                color=GT_COLOR, linewidth=0.7, alpha=0.9, label='Ground Truth')
        ax.plot(timeline, predictions[sf:ef, j],
                color=JOINT_COLORS[j], linewidth=0.9, alpha=0.85, label='Prediction')
        vj = ~np.isnan(predictions[sf:ef, j])
        if vj.any():
            ax.fill_between(timeline[vj],
                            targets[sf:ef, j][vj],
                            predictions[sf:ef, j][vj],
                            alpha=0.08, color=JOINT_COLORS[j])
        ax.text(0.99, 0.92, f"{JOINT_NAMES[j]}  R2={r2_values[j]:.4f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, fontweight='bold', color='#333')
        ax.grid(True, alpha=0.12, color='#999')
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(loc='center right', fontsize=8, frameon=False,
                      bbox_to_anchor=(0.98, 0.55))
        if j < 3: ax.set_xticklabels([])
    axes[-1].set_xlabel('Frame', fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, len(timeline) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_75', required=True)
    parser.add_argument('--meta_75', required=True)
    parser.add_argument('--pred_490', required=True)
    parser.add_argument('--meta_490', required=True)
    parser.add_argument('--out', default='figure/curves_duo.png')
    parser.add_argument('--frames_to_plot', type=int, default=300)
    args = parser.parse_args()

    t75, p75, m75 = load_data(args.pred_75, args.meta_75)
    t490, p490, m490 = load_data(args.pred_490, args.meta_490)

    r2_75 = [m75['r2'][k] for k in JOINT_KEYS]
    r2_490 = [m490['r2'][k] for k in JOINT_KEYS]

    N_75, N_490 = m75['n_frames'], m490['n_frames']
    T = 8
    n_curve = min(args.frames_to_plot, min(N_75, N_490)) if args.frames_to_plot > 0 else 0
    n_curve = max(n_curve, 1)

    sf75, ef75 = T - 1, min(N_75, T - 1 + n_curve)
    sf490, ef490 = T - 1, min(N_490, T - 1 + n_curve)
    timeline_75 = np.arange(ef75 - sf75, dtype=np.float32)
    timeline_490 = np.arange(ef490 - sf490, dtype=np.float32)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.08)

    gs_l = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.32)
    gs_r = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.32)

    axes_75 = [fig.add_subplot(gs_l[j]) for j in range(4)]
    axes_490 = [fig.add_subplot(gs_r[j]) for j in range(4)]

    # Column labels above first joint (merged into suptitle block)
    axes_75[0].text(0.02, 1.18, f"Excavator 75 (22 t)  |  R2 mean = {np.mean(r2_75):.4f}",
                    transform=axes_75[0].transAxes, fontsize=11, fontweight='bold', color='#444')
    axes_490[0].text(0.02, 1.18, f"Excavator 490 (50 t)  |  R2 mean = {np.mean(r2_490):.4f}",
                     transform=axes_490[0].transAxes, fontsize=11, fontweight='bold', color='#444')

    draw_curves(axes_75,  timeline_75,  t75,  p75,  sf75,  ef75,  r2_75)
    draw_curves(axes_490, timeline_490, t490, p490, sf490, ef490, r2_490)

    fig.suptitle("Per-joint Prediction",
                 fontsize=15, fontweight='bold', y=1.005)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
    fig.tight_layout(pad=0.5, h_pad=0.3, rect=(0, 0, 1, 0.93))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
