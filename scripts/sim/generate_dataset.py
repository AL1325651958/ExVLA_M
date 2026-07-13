"""Generate synthetic excavator VLA training data via MuJoCo simulation.

Outputs HDF5 files matching the format expected by vla_model/dataset.py:

    observations/images/main       [N, H, W, 3] uint8 BGR
    observations/images/elevation  [N, 200, 200, 3] uint8 BGR
    observations/qpos              [N, 4] float32
    action                         [N, 4] float32

Usage:
    # Generate 10 episodes for excavator 75
    python scripts/sim/generate_dataset.py --excavator 75 --episodes 10 --out_dir data/simulated/75

    # Generate for excavator 490 with custom settings
    python scripts/sim/generate_dataset.py --excavator 490 --episodes 50 --steps 200 --out_dir data/simulated/490
"""

import os
import sys
import h5py
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from scripts.sim.env import (
    ExcavatorSim,
    generate_digging_trajectory,
    generate_idle_trajectory,
    DEFAULT_JOINT_RANGE,
)


def _make_action(qpos: np.ndarray) -> np.ndarray:
    """Compute action array: action[i] ≈ qpos[i+1]. Last repeats final qpos."""
    n = len(qpos)
    action = np.zeros_like(qpos)
    action[:-1] = qpos[1:]
    action[-1] = qpos[-1]
    return action


def run_one_episode(
    sim: ExcavatorSim,
    n_steps: int,
    rng: np.random.RandomState,
    idle_prob: float = 0.1,
    episode_label: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a single episode and collect observations."""

    # Generate trajectory
    if rng.random() < idle_prob:
        traj = generate_idle_trajectory(rng, n_steps, DEFAULT_JOINT_RANGE)
        traj_type = "idle"
    else:
        traj = generate_digging_trajectory(rng, n_steps, DEFAULT_JOINT_RANGE)
        traj_type = "digging"

    sim.reset(qpos=traj[0])

    H, W = sim.img_height, sim.img_width
    E = sim.elevation_res

    mains = np.zeros((n_steps, H, W, 3), dtype=np.uint8)
    elevations = np.zeros((n_steps, E, E, 3), dtype=np.uint8)
    qpos_arr = np.zeros((n_steps, 4), dtype=np.float32)

    desc = f"  {episode_label} [{traj_type}]" if episode_label else f"  [{traj_type}]"
    iterator = tqdm(range(n_steps), desc=desc, unit="f", leave=False) \
               if HAS_TQDM else range(n_steps)

    for t in iterator:
        sim.set_qpos(traj[t])
        mains[t] = sim.render_main()
        elevations[t] = sim.compute_elevation()
        qpos_arr[t] = sim.get_qpos()

    action = _make_action(qpos_arr)
    return mains, elevations, qpos_arr, action


def save_episode_h5(out_path: str, mains: np.ndarray, elevations: np.ndarray,
                     qpos: np.ndarray, action: np.ndarray):
    """Save one episode as HDF5, matching the existing format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        obs_grp = f.create_group("observations")
        img_grp = obs_grp.create_group("images")
        img_grp.create_dataset("main", data=mains, compression="gzip",
                               compression_opts=4)
        img_grp.create_dataset("elevation", data=elevations,
                               compression="gzip", compression_opts=4)
        obs_grp.create_dataset("qpos", data=qpos)
        f.create_dataset("action", data=action)

    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"  → {Path(out_path).name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic excavator VLA data via MuJoCo simulation"
    )
    parser.add_argument("--excavator", type=str, default="75",
                        choices=["75", "490"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150,
                        help="Frames per episode (~10 fps)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--idle_prob", type=float, default=0.1,
                        help="Probability of idle episodes")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Excavator VLA Simulation Data Generator (GPU rendering)")
    print(f"  Excavator: {args.excavator}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Steps/ep:  {args.steps}")
    print(f"  Output:    {args.out_dir}")
    print(f"{'='*60}\n")

    # Build simulation
    print("Building MuJoCo model + GPU renderer...")
    import time
    t0 = time.time()
    sim = ExcavatorSim(
        excavator_type=args.excavator,
        img_width=args.img_width,
        img_height=args.img_height,
        seed=args.seed,
    )
    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"  Main camera:  {args.img_width}x{args.img_height}")
    print(f"  Elevation:    {sim.elevation_res}x{sim.elevation_res} "
          f"({sim.elevation_extent:.1f}m x {sim.elevation_extent:.1f}m, "
          f"{sim.mpp:.2f} m/px)\n")

    rng = np.random.RandomState(args.seed)
    total_frames = 0

    # Episode loop
    ep_iterator = tqdm(range(args.episodes), desc="Episodes", unit="ep") \
                  if HAS_TQDM else range(args.episodes)

    for ep in ep_iterator:
        ep_seed = args.seed + ep * 1000
        ep_rng = np.random.RandomState(ep_seed)

        n_steps = max(30, args.steps + ep_rng.randint(-20, 21))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"sim_{args.excavator}_{timestamp}_ep{ep+1:03d}.h5"
        out_path = os.path.join(args.out_dir, fname)

        sim.rebuild_terrain()

        mains, elevations, qpos_arr, action = run_one_episode(
            sim, n_steps, ep_rng,
            idle_prob=args.idle_prob,
            episode_label=f"Ep{ep+1}/{args.episodes}",
        )

        save_episode_h5(out_path, mains, elevations, qpos_arr, action)
        total_frames += n_steps

    sim.close()

    print(f"\n{'='*60}")
    print(f"  Done! {args.episodes} episodes, {total_frames} total frames")
    print(f"  Output: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
