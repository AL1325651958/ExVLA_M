"""Plot ablation comparison — per-joint per-excavator bar charts in SCI style.

Reads benchmark_results.csv and produces upright grouped-bar comparison across versions.
"""

import csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── SCI style config ──
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
EXCV_NAMES = {'75': '75 (22 t)', '490': '490 (50 t)'}
VERSION_COLORS = {
    'V9':  '#d9d9d9', 'V10': '#cce5ff', 'V11': '#e5d9f2',
    'V16': '#b3e5b0', 'V16_4': '#7fcd7f', 'V16_5': '#4db84d',
    'V17':  '#ffcc80', 'V17.1': '#ffb347', 'V17.2': '#ff8c00', 'V17.3': '#e67300',
    'V13_V3': '#99d8c9', 'V14': '#66c2a4', 'V14_2': '#2ca25f', 'V15': '#006d2c',
}

# ── Curated version list (chronological, cleaned) ──
SELECTED = [
    # STVTA
    ('V13_V3', 'STVTA\nV13', 'stvta'),
    ('V14', 'STVTA\nV14', 'stvta'),
    ('V14_2', 'STVTA\nV14₂', 'stvta'),
    ('V15', 'STVTA\nV15', 'stvta'),
    # YOLO
    ('V9',  'YOLO\nV9',  'yolo'),
    ('V10', 'YOLO\nV10', 'yolo'),
    ('V16_4', 'YOLO\nV16', 'yolo'),
    ('V17',  'YOLO\nV17',  'yolo'),
    ('V17.1','YOLO\nV17.1','yolo'),
    ('V17.3','YOLO\nV17.3','yolo'),
]

YOLO_KEY = ['yolo_checkpoint_best', 'yolo_v16_checkpoint_best', 'yolo_v17_checkpoint_best',
            'yolo_v17_1_checkpoint_best', 'yolo_v17_3_checkpoint_best',
            'stvta_v13_checkpoint_best', 'yolo_v17_1_checkpoint_best_swing',
            'yolo_v17_3_checkpoint_best_swing']
STVTA_KEY = ['stvta_checkpoint_best', 'stvta_v13_checkpoint_best']

VERSION_TO_DIR = {
    'V13_V3': 'STVTA_v13_V3', 'V14': 'STVTA_v14', 'V14_2': 'STVTA_v14_2',
    'V15': 'STVTA_v15',
    'V9':  'YOLO_ST-VLA_v9',  'V10': 'YOLO_ST-VLA_v10',
    'V16_4': 'V16_4', 'V16_5': 'V16_5',
    'V17':  'V17',   'V17.1':'V17_1',
    'V17.3':'V17_3',
}


def load_data(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            r['mae_boom'] = float(r['mae_boom']); r['mae_arm'] = float(r['mae_arm'])
            r['mae_bucket'] = float(r['mae_bucket']); r['mae_swing'] = float(r['mae_swing'])
            r['mae_mean'] = float(r['mae_mean']); r['r2_boom'] = float(r['r2_boom'])
            r['r2_arm'] = float(r['r2_arm']); r['r2_bucket'] = float(r['r2_bucket'])
            r['r2_swing'] = float(r['r2_swing']); r['r2_mean'] = float(r['r2_mean'])
            r['epoch'] = int(r['epoch']); r['n_samples'] = int(r['n_samples'])
            rows.append(r)
    return rows


def pick_best(rows, version_tag, excavator):
    """Select the best checkpoint for a version/excavator combo."""
    candidates = []
    for r in rows:
        if r['excavator'] != excavator:
            continue
        if version_tag == 'V13_V3' and r['checkpoint_dir'] == 'STVTA_v13_V3' and 'stvta_v13_checkpoint_best' in r['checkpoint_name']:
            candidates.append(r)
        elif version_tag == 'V14' and r['checkpoint_dir'] == 'STVTA_v14':
            candidates.append(r)
        elif version_tag == 'V14_2' and r['checkpoint_dir'] == 'STVTA_v14_2':
            candidates.append(r)
        elif version_tag == 'V15' and r['checkpoint_dir'] == 'STVTA_v15':
            candidates.append(r)
        elif version_tag == 'V9' and r['checkpoint_dir'] == 'YOLO_ST-VLA_v9' and 'yolo_checkpoint_best' in r['checkpoint_name']:
            candidates.append(r)
        elif version_tag == 'V10' and r['checkpoint_dir'] == 'YOLO_ST-VLA_v10':
            candidates.append(r)
        elif version_tag == 'V16_4' and r['checkpoint_dir'] == 'V16_4':
            candidates.append(r)
        elif version_tag == 'V17' and r['checkpoint_dir'] == 'V17':
            candidates.append(r)
        elif version_tag == 'V17.1' and r['checkpoint_dir'] == 'V17_1' and 'best_swing' in r['checkpoint_name']:
            candidates.append(r)
        elif version_tag == 'V17.3' and r['checkpoint_dir'] == 'V17_3' and 'best_swing' in r['checkpoint_name']:
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x['r2_mean'])


def plot_ablation(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    excavators = ['75', '490']

    for exc_idx, exc in enumerate(excavators):
        ax_r2 = axes[0][exc_idx]
        ax_mae = axes[1][exc_idx]

        versions = []
        r2_data = {j: [] for j in range(4)}
        mae_data = {j: [] for j in range(4)}
        labels = []

        for tag, label, arch in SELECTED:
            best = pick_best(rows, tag, exc)
            if best is None:
                continue
            # filter garbage
            if best['r2_mean'] < -10:
                continue
            labels.append(label)
            for j in range(4):
                r2_val = best[f'r2_{JOINT_NAMES[j].lower()}']
                r2_data[j].append(max(r2_val, -2.0))  # floor at -2 for readability
                mae_data[j].append(best[f'mae_{JOINT_NAMES[j].lower()}'])

        n = len(labels)
        x = np.arange(n)
        bar_w = 0.18
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']  # Boom=red Arm=green Bucket=blue Swing=orange

        # R² bars
        for j in range(4):
            offset = (j - 1.5) * bar_w
            bars = ax_r2.bar(x + offset, r2_data[j], bar_w, color=colors[j], alpha=0.9,
                            edgecolor='white', linewidth=0.3, label=JOINT_NAMES[j])
            # annotate values
            for bi, (bar, val) in enumerate(zip(bars, r2_data[j])):
                if val > 0.5:
                    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                              f'{val:.2f}', ha='center', va='bottom', fontsize=5, rotation=90)

        ax_r2.set_ylabel('R²', fontweight='bold')
        ax_r2.set_title(f'{EXCV_NAMES[exc]} — Coefficient of Determination', fontweight='bold', loc='left', fontsize=9)
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(labels, fontsize=6.5)
        ax_r2.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_r2.set_ylim(-0.3, 1.05)

        # MAE bars
        for j in range(4):
            offset = (j - 1.5) * bar_w
            ax_mae.bar(x + offset, mae_data[j], bar_w, color=colors[j], alpha=0.9,
                      edgecolor='white', linewidth=0.3)
        ax_mae.set_ylabel('MAE (rad)', fontweight='bold')
        ax_mae.set_title(f'{EXCV_NAMES[exc]} — Mean Absolute Error', fontweight='bold', loc='left', fontsize=9)
        ax_mae.set_xticks(x)
        ax_mae.set_xticklabels(labels, fontsize=6.5)
        ax_mae.set_ylim(0, ax_mae.get_ylim()[1])

    # Shared legend
    handles, labels_legend = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', ncol=4, fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Excavator Joint Prediction — Ablation Study', fontsize=12, fontweight='bold', y=1.08)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return fig


def plot_ablation_summary(rows, out_path):
    """Single concise figure: mean R² / mean MAE across both excavators."""
    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    mean_data = {}
    for tag, label, arch in SELECTED:
        vals_75 = pick_best(rows, tag, '75')
        vals_490 = pick_best(rows, tag, '490')
        if vals_75 is None or vals_490 is None:
            continue
        if vals_75['r2_mean'] < -10 or vals_490['r2_mean'] < -10:
            continue
        avg_r2 = (max(vals_75['r2_mean'], -2) + max(vals_490['r2_mean'], -2)) / 2
        avg_mae = (vals_75['mae_mean'] + vals_490['mae_mean']) / 2
        r2_per_joint = {}
        mae_per_joint = {}
        for j in range(4):
            jn = JOINT_NAMES[j].lower()
            r2_per_joint[jn] = (max(vals_75[f'r2_{jn}'], -2) + max(vals_490[f'r2_{jn}'], -2)) / 2
            mae_per_joint[jn] = (vals_75[f'mae_{jn}'] + vals_490[f'mae_{jn}']) / 2
        mean_data[label] = {
            'r2_mean': avg_r2, 'mae_mean': avg_mae,
            'r2_per_joint': r2_per_joint, 'mae_per_joint': mae_per_joint,
            'arch': arch,
        }

    labels = list(mean_data.keys())
    n = len(labels)
    x = np.arange(n)
    colors_joint = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    bar_w = 0.18

    # R²
    for j in range(4):
        vals = [mean_data[l]['r2_per_joint'][JOINT_NAMES[j].lower()] for l in labels]
        offset = (j - 1.5) * bar_w
        ax_r2.bar(x + offset, vals, bar_w, color=colors_joint[j], alpha=0.9,
                  edgecolor='white', linewidth=0.3)
    ax_r2.set_xticks(x); ax_r2.set_xticklabels(labels, fontsize=7)
    ax_r2.set_ylabel('R² (mean 75+490)', fontweight='bold')
    ax_r2.axhline(y=0.90, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax_r2.set_title('Mean R² by Joint', fontweight='bold', loc='left', fontsize=9)
    ax_r2.set_ylim(0.6, 1.02)

    # MAE
    for j in range(4):
        vals = [mean_data[l]['mae_per_joint'][JOINT_NAMES[j].lower()] for l in labels]
        offset = (j - 1.5) * bar_w
        ax_mae.bar(x + offset, vals, bar_w, color=colors_joint[j], alpha=0.9,
                   edgecolor='white', linewidth=0.3)
    ax_mae.set_xticks(x); ax_mae.set_xticklabels(labels, fontsize=7)
    ax_mae.set_ylabel('MAE in rad (mean 75+490)', fontweight='bold')
    ax_mae.set_title('Mean MAE by Joint', fontweight='bold', loc='left', fontsize=9)

    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.9) for c in colors_joint]
    fig.legend(handles, JOINT_NAMES, loc='upper center', ncol=4, fontsize=8, frameon=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return fig


def plot_swing_ablation(rows, out_path):
    """Detailed Swing joint analysis — R² trend across versions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    swing_r2_75, swing_r2_490 = [], []
    labels = []
    for tag, label, arch in SELECTED:
        best_75 = pick_best(rows, tag, '75')
        best_490 = pick_best(rows, tag, '490')
        if best_75 is None or best_490 is None:
            continue
        if best_75['r2_mean'] < -10:
            continue
        labels.append(label)
        swing_r2_75.append(max(best_75['r2_swing'], -2))
        swing_r2_490.append(max(best_490['r2_swing'], -2))

    x = np.arange(len(labels))
    bar_w = 0.35
    ax.bar(x - bar_w/2, swing_r2_75, bar_w, color='#3498db', alpha=0.85, label='75 (22 t)', edgecolor='white', linewidth=0.3)
    ax.bar(x + bar_w/2, swing_r2_490, bar_w, color='#e74c3c', alpha=0.85, label='490 (50 t)', edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Swing R²', fontweight='bold')
    ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Swing (Rotation) Joint — Ablation', fontweight='bold', loc='left', fontsize=10)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='benchmark_results.csv')
    parser.add_argument('--out_dir', default='figure')
    args = parser.parse_args()

    rows = load_data(args.csv)
    print(f'Loaded {len(rows)} benchmark rows')

    plot_ablation(rows, f'{args.out_dir}/ablation_per_excavator.png')
    plot_ablation_summary(rows, f'{args.out_dir}/ablation_summary.png')
    plot_swing_ablation(rows, f'{args.out_dir}/ablation_swing.png')


if __name__ == '__main__':
    main()
