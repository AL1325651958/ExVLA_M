"""Batch inference: run V17.3 on all episodes, report per-episode metrics, find best."""

import subprocess, sys, json, glob, ast, numpy as np
from pathlib import Path

DATA_DIR = Path("data/excavator-motion/data")
CHECKPOINT = "output/V17_3/yolo_v17_3_checkpoint_best.pt"
OUT_BASE = Path("export/V17_3")
DEVICE = "cuda"

results = []
for excv in ["75", "490"]:
    h5_files = sorted((DATA_DIR / excv).glob("*.h5")) + sorted((DATA_DIR / excv).glob("*.hdf5"))
    for ep in h5_files:
        ep_name = ep.stem
        out_dir = OUT_BASE / excv / ep_name
        print(f"\n{'='*60}")
        print(f"[{excv}] {ep_name}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "vla_model/export_inference.py",
            "--checkpoint", CHECKPOINT,
            "--data_path", str(ep),
            "--out_dir", str(out_dir),
            "--device", DEVICE,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            continue
        print(result.stdout)

        # Parse MAE/R² from stdout
        for line in result.stdout.split("\n"):
            if "MAE:" in line and ("R2:" in line or "R²:" in line):
                parts = line.strip().replace("R²:", "R2:").split("R2:")
                mae_str = parts[0].replace("MAE:", "").strip()
                r2_str = parts[1].strip()
                mae = ast.literal_eval(mae_str)
                r2 = ast.literal_eval(r2_str)
                results.append({
                    "excavator": excv,
                    "episode": ep_name,
                    "mae": mae,
                    "r2": r2,
                    "mae_mean": np.mean(mae),
                    "r2_mean": np.mean(r2),
                })

# ── Summary ──
print(f"\n{'='*80}")
print("SUMMARY — sorted by R² mean")
print(f"{'='*80}")
results.sort(key=lambda x: x["r2_mean"], reverse=True)
for i, r in enumerate(results):
    print(f"{i+1:2d}. [{r['excavator']}] {r['episode']:50s}  "
          f"R2={r['r2_mean']:.4f}  "
          f"MAE={r['mae_mean']:.4f}  "
          f"boom={r['r2'][0]:.4f} arm={r['r2'][1]:.4f} bucket={r['r2'][2]:.4f} swing={r['r2'][3]:.4f}")

# Save summary JSON
with open(OUT_BASE / "batch_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_BASE / 'batch_summary.json'}")
