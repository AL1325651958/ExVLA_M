"""Quick visual check of simulated HDF5 data.

Usage:
    python scripts/sim/visualize_sim_data.py --data_dir data/simulated/75 --episode 0
"""

import os
import sys
import h5py
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Visualize simulated HDF5 episodes")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to visualize (0-based)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save as MP4 instead of displaying")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Find H5 files
    files = sorted(Path(args.data_dir).glob("*.h5")) + sorted(Path(args.data_dir).glob("*.hdf5"))
    if not files:
        print(f"No HDF5 files found in {args.data_dir}")
        return
    if args.episode >= len(files):
        print(f"Episode {args.episode} out of range (max {len(files)-1})")
        return

    fpath = files[args.episode]
    print(f"Loading: {fpath}")

    with h5py.File(fpath, 'r') as f:
        mains = np.array(f['observations/images/main'])
        elevations = np.array(f['observations/images/elevation'])
        qpos = np.array(f['observations/qpos'])
        action = np.array(f['action'])

    n = len(qpos)
    print(f"Frames: {n}")
    print(f"Main shape: {mains.shape}, dtype={mains.dtype}")
    print(f"Elevation shape: {elevations.shape}, dtype={elevations.dtype}")
    print(f"qpos range: [{qpos.min(axis=0)}] → [{qpos.max(axis=0)}]")

    frames = []
    for i in range(n):
        # main is BGR, convert to RGB for display
        main_rgb = cv2.cvtColor(mains[i], cv2.COLOR_BGR2RGB)

        # elevation is BGR (colormap)
        elev_rgb = cv2.cvtColor(elevations[i], cv2.COLOR_BGR2RGB)

        # Resize elevation to match main height for side-by-side
        elev_resized = cv2.resize(elev_rgb, (main_rgb.shape[1] // 2,
                                              main_rgb.shape[0]))

        # Concatenate
        canvas = np.concatenate([main_rgb, elev_resized], axis=1)

        # Overlay joint values
        joint_text = f"swing={qpos[i,0]:.2f} boom={qpos[i,1]:.2f} arm={qpos[i,2]:.2f} bkt={qpos[i,3]:.2f}"
        cv2.putText(canvas, joint_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay frame number
        cv2.putText(canvas, f"Frame {i}/{n}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frames.append(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    if args.out:
        import imageio
        imageio.mimsave(args.out, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                        fps=args.fps)
        print(f"Saved video to {args.out}")
    else:
        # Display using OpenCV window
        print("Playing... Press ESC to exit")
        for f in frames:
            cv2.imshow("Simulated Data", f)
            key = cv2.waitKey(1000 // args.fps)
            if key == 27:  # ESC
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
