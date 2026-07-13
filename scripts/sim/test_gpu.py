"""Quick test: verify GPU simulation works."""
import sys; sys.path.insert(0, ".")
from scripts.sim.env import ExcavatorSim, generate_digging_trajectory
import numpy as np, cv2, os, time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

os.makedirs("output/sim_test", exist_ok=True)

print("Building sim (GPU renderer)..."); t0 = time.time()
sim = ExcavatorSim(excavator_type="75", img_width=640, img_height=480, seed=42)
print(f"  Done in {time.time()-t0:.1f}s")

print("Generate trajectory...")
rng = np.random.RandomState(42)
traj = generate_digging_trajectory(rng, n_steps=10)
sim.reset(qpos=traj[0])

print("Rendering 10 frames..."); t0 = time.time()
iterator = tqdm(range(10), desc="Rendering", unit="frame") if HAS_TQDM else range(10)
for i in iterator:
    sim.set_qpos(traj[i])
    main = sim.render_main()
    elev = sim.compute_elevation()
    elev_rs = cv2.resize(elev, (main.shape[1]//2, main.shape[0]))
    canvas = np.concatenate([main, elev_rs], axis=1)
    qpos = sim.get_qpos()
    cv2.putText(canvas, f"F{i} sw={qpos[0]:.2f} bm={qpos[1]:.2f} am={qpos[2]:.2f} bk={qpos[3]:.2f}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    cv2.imwrite(f"output/sim_test/frame_{i:03d}.png", canvas)

elapsed = time.time() - t0
print(f"  {elapsed:.1f}s total, {elapsed/10*1000:.0f}ms/frame")
print(f"Saved to output/sim_test/ ({len(os.listdir('output/sim_test'))} files)")
sim.close()
print("Done!")
