#!/bin/bash
# EXC Motion dataset download script
# Usage: bash scripts/download_data.sh [data_dir]
# Default data_dir: ./data/excavator-motion

set -e
DATA_DIR="${1:-./data/excavator-motion}"
BASE="https://hf-mirror.com/datasets/fuxi-robot/excavator-motion/resolve/main"

echo "============================================"
echo " EXC Motion Dataset Download"
echo " Target: $DATA_DIR"
echo " Total: 30 episodes (~24 GB)"
echo "============================================"

mkdir -p "$DATA_DIR/data/75" "$DATA_DIR/data/306" "$DATA_DIR/data/490"

download_and_verify() {
    local rel_path="$1"
    local full_path="$DATA_DIR/$rel_path"

    if [ -f "$full_path" ]; then
        python3 -c "import h5py; f=h5py.File('$full_path','r'); n=f['observations/qpos'].shape[0]; f.close(); print(f'OK: {n} frames')" 2>/dev/null && \
            { echo "  [SKIP] already OK: $rel_path"; return 0; }
        echo "  [RE-DOWNLOAD] corrupted: $rel_path"
        rm -f "$full_path"
    fi

    echo "  [DOWNLOAD] $rel_path ..."
    curl -L --retry 3 --progress-bar -o "$full_path" "$BASE/$rel_path"

    # Verify
    python3 -c "import h5py; f=h5py.File('$full_path','r'); n=f['observations/qpos'].shape[0]; f.close(); print(f'  -> OK: {n} frames')"
}

# ===== Excavator 75 (10 episodes, ~7.9 GB) =====
echo ""
echo "--- Excavator 75 ---"
for f in \
  "data/75/xcmg_data_2025-04-11-14-47-09.hdf5" \
  "data/75/xcmg_data_2025-04-11-14-59-52.hdf5" \
  "data/75/xcmg_data_2025-04-11-15-17-03.hdf5" \
  "data/75/xcmg_data_2025-04-11-15-38-07.hdf5" \
  "data/75/xcmg_data_2025-04-11-15-50-29.hdf5" \
  "data/75/xcmg_data_2025-04-11-16-05-27.hdf5" \
  "data/75/xcmg_data_2025-04-11-16-19-38.hdf5" \
  "data/75/xcmg_data_2025-04-11-16-51-51.hdf5" \
  "data/75/xcmg_data_2025-04-11-17-09-11.hdf5" \
  "data/75/xcmg_data_2025-04-11-17-46-49.hdf5"; do
    download_and_verify "$f"
done

# ===== Excavator 306 (10 episodes, ~6.5 GB) =====
echo ""
echo "--- Excavator 306 ---"
for f in \
  "data/306/2025-07-02-00-54-24_main_img_cut_1.h5" \
  "data/306/2025-07-02-01-42-20_main_img_cut_1.h5" \
  "data/306/2025-07-02-01-42-20_main_img_cut_2.h5" \
  "data/306/2025-07-02-01-42-20_main_img_cut_3.h5" \
  "data/306/2025-07-02-02-02-23_main_img_cut_1.h5" \
  "data/306/2025-07-02-02-02-23_main_img_cut_2.h5" \
  "data/306/2025-07-02-02-21-03_main_img_cut_1.h5" \
  "data/306/2025-07-02-02-42-19_main_img_cut_1.h5" \
  "data/306/2025-07-02-02-42-19_main_img_cut_2.h5" \
  "data/306/2025-07-02-03-02-52_main_img_cut_3.h5"; do
    download_and_verify "$f"
done

# ===== Excavator 490 (10 episodes, ~9.7 GB) =====
echo ""
echo "--- Excavator 490 ---"
for f in \
  "data/490/robot_data_20250420_123326.h5" \
  "data/490/robot_data_20250420_135018.h5" \
  "data/490/robot_data_20250420_142113.h5" \
  "data/490/robot_data_20250420_164956.h5" \
  "data/490/robot_data_20250420_170112.h5" \
  "data/490/robot_data_20250423_133123.h5" \
  "data/490/robot_data_20250423_134505.h5" \
  "data/490/robot_data_20250423_140312.h5" \
  "data/490/robot_data_20250430_121917.h5" \
  "data/490/robot_data_20250430_123028.h5"; do
    download_and_verify "$f"
done

echo ""
echo "============================================"
echo " Download complete! Running final check..."
echo "============================================"
python3 -c "
import h5py, os
for exc in ['75','306','490']:
    d = f'$DATA_DIR/data/{exc}'
    files = sorted(os.listdir(d)) if os.path.exists(d) else []
    ok, total_frames = 0, 0
    for fname in files:
        try:
            with h5py.File(os.path.join(d, fname), 'r') as f:
                n = f['observations/qpos'].shape[0]
            ok += 1; total_frames += n
        except: pass
    print(f'{exc}: {ok}/{len(files)} OK, {total_frames} frames')
"
echo ""
echo "Done! Run training with:"
echo "  python vla_model/train.py --batch_size 4 --epochs 80"
