"""Excavator-STVTA V12 visualization: dual-modality masks + per-joint fusion alpha."""

import sys, argparse
from pathlib import Path
import numpy as np, torch, cv2, imageio, h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model_stvta import ExcavatorSTVTA
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

JOINT_NAMES = ['Boom', 'Arm', 'Bucket', 'Swing']
MODALITY_NAMES = ['RGB', 'Elevation']
REGION_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]
GT_COLOR, PRED_COLOR = '#333333', '#e74c3c'
MAIN_W, MAIN_H, MASK_W, MASK_H = 270, 270, 135, 135
CURVE_W, CURVE_H_PER_JOINT = 900, 100
PAD = 4


def preprocess_image(img_bgr, size=224):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def resize_keep_aspect(img, t_w, t_h):
    h, w = img.shape[:2]
    r = min(t_h / h, t_w / w)
    nh, nw = int(h * r), int(w * r)
    img = cv2.resize(img, (nw, nh))
    pt, pb = (t_h - nh) // 2, t_h - nh - (t_h - nh) // 2
    pl, pr = (t_w - nw) // 2, t_w - nw - (t_w - nw) // 2
    return cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def render_mask(bgr, mask, ci):
    h, w = bgr.shape[:2]
    ov = bgr.copy().astype(np.float32)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0, 1)
    if mask.max() > mask.min(): mask = (mask - mask.min()) / (mask.max() - mask.min())
    c = np.array(REGION_COLORS[ci], dtype=np.float32)
    alpha = 0.45
    for ch in range(3): ov[:, :, ch] = ov[:, :, ch] * (1 - alpha * mask) + c[ch] * (alpha * mask)
    return np.clip(ov, 0, 255).astype(np.uint8)


def render_curves(timeline, targets, preds, cur_idx, range_frames=200):
    N = len(timeline)
    half = range_frames // 2
    s, e = max(0, cur_idx - half), min(N, cur_idx + half)
    ts = timeline[s:e]
    fig, axes = plt.subplots(4, 1, figsize=(CURVE_W / 100, CURVE_H_PER_JOINT * 4 / 100), dpi=100)
    fig.patch.set_facecolor('white')
    for j in range(4):
        ax = axes[j]; ax.set_facecolor('white'); ax.tick_params(labelsize=6)
        tp = timeline[:cur_idx + 1]
        ax.plot(ts, targets[s:e, j], color='#cccccc', linewidth=0.6, alpha=0.7)
        ax.plot(tp[s:], targets[:cur_idx + 1, j][s:], color=GT_COLOR, linewidth=1.0, label='GT')
        ax.plot(tp[s:], preds[:cur_idx + 1, j][s:], color=PRED_COLOR, linewidth=1.0, label='Pred')
        ax.axvline(x=timeline[cur_idx], color='#3498db', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.plot(timeline[cur_idx], targets[cur_idx, j], 'o', color=GT_COLOR, markersize=4)
        ax.plot(timeline[cur_idx], preds[cur_idx, j], 'o', color=PRED_COLOR, markersize=4)
        ax.set_xlim(ts[0], ts[-1]); ax.set_ylabel(JOINT_NAMES[j], fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.2); [ax.spines[k].set_visible(False) for k in ['top', 'right']]
        if j == 0: ax.legend(loc='upper right', fontsize=7)
        if j < 3: ax.set_xticklabels([])
    axes[-1].set_xlabel('Frame', fontsize=8)
    fig.tight_layout(pad=0.5, h_pad=0.2)
    c = FigureCanvasAgg(fig); c.draw()
    buf = np.asarray(c.buffer_rgba())[:, :, :3]; plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Visualize Excavator-STVTA V12")
    parser.add_argument("--checkpoint", type=str, default="output/YOLO_ST-VLA_v12/stvta_checkpoint_best.pt")
    parser.add_argument("--data_path", type=str, default="data/excavator-motion/data/75/xcmg_data_2025-04-11-17-46-49.hdf5")
    parser.add_argument("--out_dir", type=str, default="output/stvta_vis")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_masks", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load checkpoint ──
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    sd_keys = set(sd.keys())

    # Auto-detect config
    hidden_dim = sd.get("rgb_branch.grid_proj.weight").shape[0]
    # Count layers
    n_layers = 4
    for k in sd_keys:
        if k.startswith("rgb_branch.encoder.layers."):
            idx = int(k.split(".")[3])
            n_layers = max(n_layers, idx + 1)
    ff_dim = sd.get("rgb_branch.encoder.layers.0.linear1.weight").shape[0]

    is_v13 = any("joint_relation" in k or "vel_aux_head" in k for k in sd_keys)
    version_tag = "V13" if is_v13 else "V12"
    print(f"  {version_tag} detected: hidden_dim={hidden_dim}, n_layers={n_layers}, ff_dim={ff_dim}")
    G = args.img_size // 16

    model = ExcavatorSTVTA(
        seq_len=args.seq_len, img_size=args.img_size,
        hidden_dim=hidden_dim, n_heads=8,
        n_layers=n_layers, ff_dim=ff_dim, dropout=0.0, pretrained=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ── Load data ──
    print(f"Loading: {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        targets = np.zeros_like(qpos)
        targets[:-1] = qpos[1:]; targets[-1] = qpos[-1]
    N, T = len(targets), args.seq_len

    # Parse excavator ID
    pl = args.data_path.lower()
    if '/75/' in pl or '\\75\\' in pl: excv_id = 0
    elif '/306/' in pl or '\\306\\' in pl: excv_id = 1
    elif '/490/' in pl or '\\490\\' in pl: excv_id = 2
    else: excv_id = 3
    excv_t = torch.tensor([excv_id], dtype=torch.long).to(device)

    # ── Preprocess ──
    print("Preprocessing...")
    rgb_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
    elev_pp = np.zeros((N, 3, args.img_size, args.img_size), dtype=np.float32)
    for i in tqdm(range(N), desc="  Preprocessing"):
        rgb_pp[i] = preprocess_image(mains[i], args.img_size)
        elev_pp[i] = preprocess_image(elevations[i], args.img_size)

    # ── Inference ──
    print("Running inference...")
    predictions = np.full((N, 4), np.nan, dtype=np.float32)
    all_masks = np.zeros((N, 2, 4, G, G), dtype=np.float32)
    all_alphas = np.zeros((N, 4), dtype=np.float32)
    predictions[:T - 1] = targets[:T - 1]

    for start in tqdm(range(0, N - T), desc="  Inference"):
        end = start + T
        rgb_s = torch.from_numpy(rgb_pp[start:end]).unsqueeze(0).to(device)
        elev_s = torch.from_numpy(elev_pp[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(rgb_s, elev_s, excavator_id=excv_t, return_diagnostics=True)
            action = out[0]; masks_spatial = out[2]; alpha = out[3]  # out[4] is mask_stats dict
        tgt_idx = start + T - 1
        predictions[tgt_idx] = model.decode_action(action)[0].cpu().numpy()
        all_masks[tgt_idx] = masks_spatial.mean(dim=3)[0].cpu().numpy()
        all_alphas[tgt_idx] = alpha[0].cpu().numpy()

    # ── MAE ──
    valid = ~np.isnan(predictions[:, 0])
    mae = np.abs(predictions[valid] - targets[valid]).mean(axis=0)
    print(f"\nMAE: Boom={mae[0]:.4f} Arm={mae[1]:.4f} Bucket={mae[2]:.4f} Swing={mae[3]:.4f}")
    print(f"  Mean: {mae.mean():.4f} rad = {mae.mean()*57.3:.2f}°")

    # ── Render ──
    print("Rendering...")
    timeline = np.arange(N, dtype=np.float32)
    # Layout: [MainRGB | M0rgb M0elev | M1rgb M1elev]
    #         [Elev    | M2rgb M2elev | M3rgb M3elev]
    #         [alpha bar | Curves]
    total_w = MAIN_W + 4 * MASK_W
    title_h = 30; alpha_h = 20
    frames = []

    for i in tqdm(range(T - 1, N), desc="  Rendering"):
        main_rgb = cv2.cvtColor(mains[i], cv2.COLOR_BGR2RGB)
        main_rgb = resize_keep_aspect(main_rgb, MAIN_W, MAIN_H)
        elev_v = cv2.cvtColor(elevations[i], cv2.COLOR_BGR2RGB)
        elev_v = resize_keep_aspect(elev_v, MAIN_W, MAIN_H)

        # Mask panels: for each joint, show [rgb_mask | elev_mask]
        mask_panels = []
        rgb_small = cv2.resize(mains[i], (MASK_W, MASK_H))
        elev_small = cv2.resize(elevations[i], (MASK_W, MASK_H))
        for j in range(4):
            m_rgb = render_mask(rgb_small, all_masks[i, 0, j], j)
            m_elev = render_mask(elev_small, all_masks[i, 1, j], j)
            panel = np.concatenate([m_rgb, m_elev], axis=0)
            cv2.putText(panel, f"{JOINT_NAMES[j]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            mask_panels.append(panel)

        top_row = np.concatenate([main_rgb] + mask_panels[:2], axis=1)
        bot_row = np.concatenate([elev_v] + mask_panels[2:], axis=1)
        top_section = np.concatenate([top_row, bot_row], axis=0)
        tw = top_section.shape[1]

        # Title
        title = np.full((title_h, tw, 3), 255, dtype=np.uint8)
        cv2.putText(title, f"Frame {i}/{N} | MAE Boom={mae[0]:.4f} Arm={mae[1]:.4f} Bucket={mae[2]:.4f} Swing={mae[3]:.4f}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

        # Alpha bar
        alpha_bar = np.full((alpha_h, tw, 3), 255, dtype=np.uint8)
        alphas = all_alphas[i]
        for j in range(4):
            x0 = 10 + j * 100; x1 = x0 + 90
            cv2.putText(alpha_bar, f"{JOINT_NAMES[j]}:{alphas[j]:.2f}", (x0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)

        # Curves
        curves = render_curves(timeline, targets, predictions, i, 200)
        ch, cw = curves.shape[:2]
        if cw != tw: curves = cv2.resize(curves, (tw, ch))

        frame = np.concatenate([title, top_section, alpha_bar, curves], axis=0)
        frames.append(frame)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{args.out_dir}/{Path(args.data_path).stem}_stvta_v12.mp4"
    imageio.mimsave(out_path, frames, fps=args.fps, macro_block_size=1)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
