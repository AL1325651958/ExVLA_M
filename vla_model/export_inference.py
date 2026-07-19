"""Inference export: run model on one episode, save predictions & masks to CSV+NPY.

Outputs per output_dir:
  predictions.csv  — frame, target_boom, target_arm, target_bucket, target_swing,
                       pred_boom,   pred_arm,   pred_bucket,   pred_swing
  all_masks.npy    — [N, 2, 4, 14, 14] (modality × joint × spatial)
  meta.json        — checkpoint path, version, excavator, episode path, per-joint MAE/R2
"""

import sys, argparse, json, h5py, numpy as np, torch, csv, cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vla_model.model_yolo import (
    ExcavatorVLAYolo, load_compatible_state_dict,
    upgrade_legacy_v17_1_state_dict,
)
from vla_model.dataset import IMAGENET_MEAN, IMAGENET_STD

JOINT_NAMES = ['boom', 'arm', 'bucket', 'swing']


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
        return True, "v17.1"
    has_v17 = any("joint_logit_bias" in k for k in state_keys)
    has_temp = any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys)
    if has_v17 and has_temp: return True, "v17.1"
    if has_v17: return True, "v17"
    if any("rgb_proj" in k or "cross_rgb_from_elev" in k for k in state_keys):
        return True, "v16"
    if any("temporal_mask_mixer" in k or "pose_aux_head" in k for k in state_keys): return False, "v10"
    return False, "v9"


def infer_config(state_dict):
    attn = state_dict.get("encoder.layers.0.self_attn.in_proj_weight",
                          state_dict.get("encoder.layers.0.linear1.weight"))
    layers = {int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")}
    return attn.shape[1], max(layers) + 1 if layers else 4, \
           state_dict.get("encoder.layers.0.linear1.weight", attn).shape[0]


def compute_metrics(predictions, targets):
    valid = ~np.isnan(predictions[:, 0])
    pred, gt = predictions[valid], targets[valid]
    n = len(pred)
    mae = np.zeros(4)
    for j in range(3):
        mae[j] = np.abs(pred[:, j] - gt[:, j]).mean()
    mae[3] = np.abs(_circular_error(pred[:, 3], gt[:, 3])).mean()
    r2 = np.zeros(4)
    for j in range(3):
        ss_res = ((pred[:, j] - gt[:, j]) ** 2).sum()
        ss_tot = max(((gt[:, j] - gt[:, j].mean()) ** 2).sum(), 1e-10)
        r2[j] = 1 - ss_res / ss_tot
    se = _circular_error(pred[:, 3], gt[:, 3])
    s = gt[:, 3]; ma = np.arctan2(np.sin(s).mean(), np.cos(s).mean())
    ctd = _circular_error(s, ma); st = max((ctd ** 2).sum(), 1e-10)
    r2[3] = 1 - (se ** 2).sum() / st
    return mae, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out_dir', default='export')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──
    print(f'Loading: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    is_v16, version_arg = detect_version(set(sd.keys()), ckpt.get('model_version'))
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
    print(f'  version={version_arg}  n_layers={n_layers}  hidden_dim={hidden_dim}')

    # ── Load episode ──
    print(f'Loading: {args.data_path}')
    with h5py.File(args.data_path, 'r') as f:
        mains = f['observations/images/main'][:]
        elevations = f['observations/images/elevation'][:]
        qpos = f['observations/qpos'][:].astype(np.float32)
        targets = np.zeros_like(qpos); targets[:-1] = qpos[1:]; targets[-1] = qpos[-1]
    N, T = len(targets), 8
    print(f'  {N} frames')

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

    # ── Export CSV ──
    ep_name = Path(args.data_path).stem
    csv_path = out_dir / f'{ep_name}_predictions.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame'] +
                   [f'target_{jn}' for jn in JOINT_NAMES] +
                   [f'pred_{jn}' for jn in JOINT_NAMES])
        for i in range(N):
            w.writerow([i] +
                       [f'{targets[i, j]:.6f}' for j in range(4)] +
                       [f'{predictions[i, j]:.6f}' if not np.isnan(predictions[i, j]) else 'nan'
                        for j in range(4)])
    print(f'Saved: {csv_path}')

    # ── Export masks ──
    npy_path = out_dir / f'{ep_name}_masks.npy'
    np.save(npy_path, all_masks)
    print(f'Saved: {npy_path}  shape={all_masks.shape}')

    # ── Export meta ──
    mae, r2 = compute_metrics(predictions, targets)
    meta = {
        'checkpoint': args.checkpoint,
        'version': version_arg,
        'data_path': args.data_path,
        'excavator': {0: '75', 1: '306', 2: '490'}[excv_id],
        'n_frames': N,
        'seq_len': T,
        'grid_size': G,
        'mae': {JOINT_NAMES[j]: float(mae[j]) for j in range(4)},
        'r2': {JOINT_NAMES[j]: float(r2[j]) for j in range(4)},
        'mae_mean': float(mae.mean()),
        'r2_mean': float(r2.mean()),
    }
    meta_path = out_dir / f'{ep_name}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved: {meta_path}')
    print(f'  MAE: {[f"{m:.4f}" for m in mae]}  R2: {[f"{r:.4f}" for r in r2]}')


if __name__ == '__main__':
    main()
