"""Render simulation frames for visual inspection.

Usage:
    python scripts/sim/preview.py                           # default: 10 frames, show window
    python scripts/sim/preview.py --save output/preview/    # save as PNG frames
    python scripts/sim/preview.py --save output/test.mp4    # save as MP4 video
    python scripts/sim/preview.py --steps 30 --excavator 490
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import cv2
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from scripts.sim.env import ExcavatorSim, generate_digging_trajectory


def main():
    parser = argparse.ArgumentParser(description="Preview simulated excavator data")
    parser.add_argument("--excavator", type=str, default="75", choices=["75", "490"])
    parser.add_argument("--steps", type=int, default=10, help="Number of frames")
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Save path: directory for PNGs, or .mp4 file for video")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video output")
    args = parser.parse_args()

    t_start = time.time()
    print(f"Building {args.excavator} excavator simulation...")
    sim = ExcavatorSim(
        excavator_type=args.excavator,
        img_width=args.img_width,
        img_height=args.img_height,
        seed=args.seed,
    )
    print(f"  Camera: {args.img_width}x{args.img_height}")
    print(f"  Elevation map: {sim.elevation_res}x{sim.elevation_res} "
          f"({sim.elevation_extent:.1f}m x {sim.elevation_extent:.1f}m)")

    rng = np.random.RandomState(args.seed)
    traj = generate_digging_trajectory(rng, n_steps=args.steps)
    sim.reset(qpos=traj[0])

    frames = []
    iterator = tqdm(enumerate(traj), total=len(traj), desc="Rendering", unit="f") \
               if HAS_TQDM else enumerate(traj)

    t_render = time.time()
    for i, tgt in iterator:
        sim.set_qpos(tgt)
        main_img = sim.render_main()          # BGR, [H, W, 3]
        elev_img = sim.compute_elevation()    # BGR, [200, 200, 3]

        # Resize elevation to match main height, preserve aspect
        elev_h = main_img.shape[0]
        elev_w = int(elev_img.shape[1] * elev_h / elev_img.shape[0])
        elev_rs = cv2.resize(elev_img, (elev_w, elev_h))

        # Side-by-side: main | elevation
        canvas = np.concatenate([main_img, elev_rs], axis=1)

        # Overlay joint info
        qpos = sim.get_qpos()
        cv2.putText(canvas, f'sw={qpos[0]:.2f} bm={qpos[1]:.2f} am={qpos[2]:.2f} bk={qpos[3]:.2f}',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, f'Step {i+1}/{args.steps}',
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        frames.append(canvas)

    t_render_elapsed = time.time() - t_render
    sim.close()

    print(f"  Render: {t_render_elapsed:.1f}s ({t_render_elapsed/args.steps*1000:.0f} ms/frame)")

    # ── Output ──
    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == '.mp4':
            import imageio
            print("Encoding MP4...")
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(save_path), rgb_frames, fps=args.fps)
            print(f"  Saved video: {save_path}  ({len(frames)} frames, {args.fps} fps)")
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(tqdm(frames, desc="Writing PNGs") if HAS_TQDM else frames):
                cv2.imwrite(str(save_path / f'frame_{i:03d}.png'), f)
            print(f"  Saved {len(frames)} frames → {save_path}/")
    else:
        # Interactive display
        print("\nPlaying... Press ESC to exit, SPACE to pause")
        paused = False
        i = 0
        while True:
            cv2.imshow("Simulation Preview (Left: RGB Main | Right: Elevation)", frames[i])
            key = cv2.waitKey(0 if paused else int(1000 / args.fps)) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d') or key == 83:  # right arrow
                i = min(i + 1, len(frames) - 1)
            elif key == ord('a') or key == 81:  # left arrow
                i = max(i - 1, 0)
            else:
                i = (i + 1) % len(frames) if not paused else i
        cv2.destroyAllWindows()

    print(f"  Total elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
