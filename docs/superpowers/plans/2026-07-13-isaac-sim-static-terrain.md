# Isaac Sim Static-Terrain Excavator Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate custom articulated-excavator VLA episodes in Isaac Sim with RGB, metric height maps, depth, and an HDF5 contract compatible with the existing dataset reader.

**Architecture:** Keep Isaac-specific scene construction under `scripts/isaac_sim/` and keep coordinate conversion, height visualization, and HDF5 validation in importable NumPy modules. A top-down calibrated camera provides depth that is converted into a metric height raster; the required VLA datasets retain their existing names and BGR conventions.

**Tech Stack:** Isaac Sim Python API, USD/PhysX articulations, Replicator camera annotators, NumPy, h5py, OpenCV, unittest.

---

### Task 1: Add dependency-free metric height-map utilities

**Files:**
- Create: `scripts/isaac_sim/__init__.py`
- Create: `scripts/isaac_sim/heightmap.py`
- Create: `tests/test_isaac_heightmap.py`

- [ ] **Step 1: Write failing tests for depth-to-height conversion and BGR visualization**

```python
import unittest
import numpy as np
from scripts.isaac_sim.heightmap import depth_to_height, colorize_height


class HeightMapTests(unittest.TestCase):
    def test_depth_to_height_uses_top_camera_height(self):
        depth = np.array([[2.0, 5.0], [np.inf, 0.0]], dtype=np.float32)
        height = depth_to_height(depth, camera_z=10.0, invalid_height=0.0)
        np.testing.assert_allclose(height, [[8.0, 5.0], [0.0, 0.0]])

    def test_colorize_height_returns_bgr_uint8(self):
        image = colorize_height(np.array([[0.0, 2.0]], dtype=np.float32))
        self.assertEqual(image.shape, (1, 2, 3))
        self.assertEqual(image.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: FAIL with `ModuleNotFoundError` because `scripts.isaac_sim.heightmap` does not exist.

- [ ] **Step 3: Implement the minimal pure-NumPy utilities**

```python
def depth_to_height(depth_m, camera_z, invalid_height=0.0):
    depth = np.asarray(depth_m, dtype=np.float32)
    height = np.full(depth.shape, invalid_height, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    height[valid] = camera_z - depth[valid]
    return height


def colorize_height(height_m):
    finite = np.isfinite(height_m)
    lo = float(height_m[finite].min()) if finite.any() else 0.0
    hi = float(height_m[finite].max()) if finite.any() else lo + 1.0
    normalized = np.clip((height_m - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/isaac_sim/__init__.py scripts/isaac_sim/heightmap.py tests/test_isaac_heightmap.py
git commit -m "feat: add metric height-map utilities"
```

### Task 2: Build the parameterized Isaac Sim scene and calibrated cameras

**Files:**
- Create: `scripts/isaac_sim/scene.py`
- Create: `scripts/isaac_sim/terrain.py`
- Create: `tests/test_isaac_heightmap.py`

- [ ] **Step 1: Add a failing terrain contract test**

```python
from scripts.isaac_sim.terrain import sample_height_field

def test_height_field_is_finite_and_nonflat():
    height = sample_height_field(resolution=200, extent_m=20.0, seed=7)
    assert height.shape == (200, 200)
    assert np.isfinite(height).all()
    assert float(height.max() - height.min()) > 0.1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: FAIL because `sample_height_field` does not exist.

- [ ] **Step 3: Implement terrain and scene builders**

Implement `sample_height_field` as a deterministic sum of Gaussian mounds and a shallow trench, clipped to `[-0.5, 2.0]` metres. In `scene.py`, expose:

```python
def build_episode_scene(world, seed, terrain_extent_m=20.0, height_resolution=200):
    """Return articulation handles, camera handles, height field, and metadata."""

def set_joint_positions(articulation, qpos):
    """Apply [swing, boom, arm, bucket] targets in radians."""
```

Create the excavator from named USD primitives `base`, `upper`, `cab`, `boom`, `arm`, and `bucket`; attach four revolute joints named `swing`, `boom`, `arm`, and `bucket`. Generate a PhysX height-field mesh from the sampled raster. Add a perspective RGB camera aimed at `[2, 0, 1]` and an orthographic top camera at `[0, 0, 15]` looking downward over the 20-metre square work area.

- [ ] **Step 4: Run the pure-Python terrain tests**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: PASS without requiring Isaac Sim.

- [ ] **Step 5: Commit**

```bash
git add scripts/isaac_sim/scene.py scripts/isaac_sim/terrain.py tests/test_isaac_heightmap.py
git commit -m "feat: add Isaac Sim excavator scene"
```

### Task 3: Generate compatible HDF5 episodes from Isaac sensors

**Files:**
- Create: `scripts/isaac_sim/generate_dataset.py`
- Create: `scripts/isaac_sim/hdf5_io.py`
- Modify: `tests/test_isaac_heightmap.py`

- [ ] **Step 1: Write a failing HDF5 contract test**

```python
import tempfile
import h5py
from scripts.isaac_sim.hdf5_io import save_episode

def test_saved_episode_preserves_vla_required_paths():
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        save_episode(tmp.name, main_bgr=np.zeros((2, 4, 4, 3), np.uint8),
                     elevation_bgr=np.zeros((2, 3, 3, 3), np.uint8),
                     qpos=np.zeros((2, 4), np.float32),
                     height_m=np.zeros((2, 3, 3), np.float32),
                     depth_m=np.ones((2, 3, 3), np.float32), metadata={"seed": 1})
        with h5py.File(tmp.name) as f:
            assert f["observations/images/main"].shape == (2, 4, 4, 3)
            assert f["observations/images/elevation"].shape == (2, 3, 3, 3)
            assert f["observations/qpos"].shape == (2, 4)
            assert f["action"].shape == (2, 4)
            assert f["observations/sim/height_m"].shape == (2, 3, 3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: FAIL because `hdf5_io.save_episode` does not exist.

- [ ] **Step 3: Implement HDF5 writing and generator CLI**

Implement `save_episode` with `action[:-1] = qpos[1:]` and repeat the final qpos for the final action. Store `height_m`, `depth_m`, intrinsic matrix, extrinsic matrix, and JSON metadata in `observations/sim`. Implement a CLI accepting `--episodes`, `--steps`, `--out_dir`, `--seed`, `--height_res`, and `--extent_m`. Launch Isaac with `SimulationApp({"headless": True})`, render both cameras each frame, convert RGB to BGR, validate finite/non-flat heights and non-empty RGB, then save `episode_00000.h5` through `episode_XXXXX.h5`.

- [ ] **Step 4: Run HDF5 tests**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/isaac_sim/generate_dataset.py scripts/isaac_sim/hdf5_io.py tests/test_isaac_heightmap.py
git commit -m "feat: generate Isaac Sim VLA episodes"
```

### Task 4: Add preview, server instructions, and an Isaac smoke test

**Files:**
- Create: `scripts/isaac_sim/preview.py`
- Create: `scripts/isaac_sim/README.md`
- Create: `scripts/isaac_sim/test_headless.py`

- [ ] **Step 1: Write a failing preview-layout test**

```python
from scripts.isaac_sim.preview import compose_preview

def test_preview_layout_contains_rgb_and_height_map():
    frame = compose_preview(np.zeros((20, 30, 3), np.uint8),
                            np.zeros((20, 20, 3), np.uint8),
                            np.zeros(4, np.float32))
    assert frame.shape == (20, 50, 3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: FAIL because `compose_preview` does not exist.

- [ ] **Step 3: Implement preview and server documentation**

Implement `compose_preview(main_bgr, elevation_bgr, qpos)` by resizing the elevation image to the main image height and concatenating the two images after adding the qpos overlay. `preview.py` must load one generated episode and write an MP4 via `imageio`. `test_headless.py` must launch one headless Isaac scene, render one RGB/depth pair, assert that dimensions are nonzero and close the app in `finally`. `README.md` must include the exact Isaac Sim Python invocation:

```bash
./python.sh scripts/isaac_sim/test_headless.py
./python.sh scripts/isaac_sim/generate_dataset.py --episodes 10 --steps 200 --out_dir data/isaac_sim
./python.sh scripts/isaac_sim/preview.py --episode data/isaac_sim/episode_00000.h5 --out output/isaac_preview.mp4
```

- [ ] **Step 4: Run local tests and the server smoke test**

Run: `python -m unittest tests.test_isaac_heightmap -v`

Expected: PASS.

Run on Isaac server: `./python.sh scripts/isaac_sim/test_headless.py`

Expected: exits with `Headless Isaac sensor smoke test passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/isaac_sim/preview.py scripts/isaac_sim/test_headless.py scripts/isaac_sim/README.md tests/test_isaac_heightmap.py
git commit -m "docs: add Isaac Sim data generation workflow"
```

## Self-review

- Spec coverage: Tasks 1-2 implement metric height maps, custom scene, terrain, and camera geometry; Task 3 preserves the VLA HDF5 contract and adds sensor metadata; Task 4 provides visual validation and headless server operation.
- Placeholder scan: no unresolved implementation placeholders remain.
- Type consistency: height/depth are float32 `[N,H,W]`, required images are BGR uint8 `[N,H,W,3]`, and qpos/action are float32 `[N,4]` throughout.
