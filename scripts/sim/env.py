"""Excavator simulator — MuJoCo physics + GPU OpenGL rendering.

Generates RGB images via MuJoCo's built-in GPU-accelerated Renderer.
Elevation maps computed procedurally (fast 2D array operations).
Outputs HDF5 matching the existing VLA training format.
"""

import numpy as np
import cv2
import mujoco
from pathlib import Path
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# Excavator physical parameters
# ---------------------------------------------------------------------------
EXCAVATOR_PARAMS = {
    "75": {
        "boom_len": 3.6957, "arm_len": 1.62233,
        "bucket_len": 0.8, "bucket_width": 0.6,
        "operating_arm_height": 1.4, "meter_per_pixel": 0.1,
        "base_radius": 2.0, "cab_width": 1.5, "cab_height": 1.2,
        "boom_init_angle": 1.0, "arm_init_angle": 1.5,
    },
    "490": {
        "boom_len": 6.670, "arm_len": 2.90746,
        "bucket_len": 1.5, "bucket_width": 1.2,
        "operating_arm_height": 2.46, "meter_per_pixel": 0.2,
        "base_radius": 3.0, "cab_width": 2.0, "cab_height": 2.0,
        "boom_init_angle": 1.0, "arm_init_angle": 1.5,
    },
}

DEFAULT_JOINT_RANGE = np.array([
    [-3.14, 3.14], [-0.8, 1.2], [-1.8, 0.5], [-1.0, 2.5],
])

TERRAIN_COLORMAP = np.array([
    [0.2, 0.4, 0.8], [0.3, 0.6, 0.3],
    [0.6, 0.5, 0.2], [0.85, 0.8, 0.7],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# MJCF XML builder — terrain mounds as box geoms + camera definitions
# ---------------------------------------------------------------------------

def _build_mjcf(params: dict, joint_range: np.ndarray,
                mounds: List[Tuple], rng: np.random.RandomState) -> str:
    """Generate MuJoCo MJCF XML with terrain mounds and cameras."""

    bl, al = params["boom_len"], params["arm_len"]
    bu, bw = params["bucket_len"], params["bucket_width"]
    oah, br = params["operating_arm_height"], params["base_radius"]
    cw, ch = params["cab_width"], params["cab_height"]

    mound_geoms = ""
    for i, (cx, cy, sx, sy, h) in enumerate(mounds):
        rv = 0.4 + rng.uniform(-0.08, 0.08)
        gv = 0.3 + rng.uniform(-0.05, 0.05)
        bv = 0.18 + rng.uniform(-0.05, 0.05)
        mound_geoms += (
            f'\n      <geom name="mound_{i}" type="box" '
            f'size="{sx:.2f} {sy:.2f} {h*0.5:.2f}" '
            f'pos="{cx:.2f} {cy:.2f} {h*0.5:.2f}" '
            f'rgba="{rv:.2f} {gv:.2f} {bv:.2f} 1"/>'
        )

    # Build small reference markers (cones/boxes) so the scene isn't empty at distance
    extra_props = ""
    for i in range(rng.randint(2, 5)):
        px = rng.uniform(-10, 10)
        py = rng.uniform(-10, 10)
        extra_props += (
            f'\n      <geom name="prop_{i}" type="cylinder" '
            f'size="0.15 0.02" pos="{px:.1f} {py:.1f} 0.02" '
            f'rgba="0.5 0.4 0.3 1"/>'
        )

    return f"""<mujoco model="excavator">
  <compiler angle="radian"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <map znear="0.05" zfar="80"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 1.0" rgb2="0.9 0.95 1.0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.3 0.35 0.25" rgb2="0.38 0.42 0.3"/>
    <material name="ground_mat" texture="grid" texrepeat="12 12" reflectance="0.15"/>
    <material name="base_mat" rgba="0.22 0.22 0.28 1.0"/>
    <material name="cab_mat" rgba="0.88 0.72 0.15 1.0"/>
    <material name="boom_mat" rgba="0.72 0.65 0.18 1.0"/>
    <material name="arm_mat" rgba="0.62 0.55 0.20 1.0"/>
    <material name="bucket_mat" rgba="0.40 0.40 0.40 1.0"/>
  </asset>

  <worldbody>
    <light name="sun" pos="8 12 20" dir="-1 -1 -1" diffuse="1.0 1.0 0.95" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="-10 -6 6" dir="0.5 -0.5 -0.5" diffuse="0.4 0.4 0.4" specular="0 0 0"/>

    <body name="ground" pos="0 0 0">
      <geom type="plane" size="30 30 0.1" material="ground_mat"/>{mound_geoms}{extra_props}
    </body>

    <body name="base" pos="0 0 0.2">
      <geom name="base_geom" type="cylinder" size="{br} 0.3" material="base_mat"/>
      <geom name="base_top" type="box" size="{br} {br*0.6} 0.12" pos="0 0 0.38" material="base_mat"/>
      <body name="cab" pos="0 0 0.5">
        <joint name="swing" type="hinge" axis="0 0 1"
               range="{joint_range[0,0]} {joint_range[0,1]}" damping="2.0"/>
        <geom name="cab_body" type="box" size="{cw*0.4} {cw*0.5} {ch*0.45}"
              pos="0 0 {ch*0.45}" material="cab_mat"/>
        <body name="boom_base" pos="0 0 {oah}">
          <joint name="boom" type="hinge" axis="0 -1 0"
                 range="{joint_range[1,0]} {joint_range[1,1]}" damping="4.0"/>
          <geom name="boom_geom" type="box" size="{bl*0.5} 0.13 0.13"
                pos="{bl*0.5} 0 0" material="boom_mat"/>
          <body name="arm_base" pos="{bl} 0 0">
            <joint name="arm" type="hinge" axis="0 -1 0"
                   range="{joint_range[2,0]} {joint_range[2,1]}" damping="2.0"/>
            <geom name="arm_geom" type="box" size="{al*0.5} 0.11 0.11"
                  pos="{al*0.5} 0 0" material="arm_mat"/>
            <body name="bucket_base" pos="{al} 0 0">
              <joint name="bucket" type="hinge" axis="0 -1 0"
                     range="{joint_range[3,0]} {joint_range[3,1]}" damping="1.0"/>
              <geom name="bucket_body" type="box" size="{bu*0.5} {bw*0.5} {bu*0.28}"
                    pos="{bu*0.48} 0 {-bu*0.18}" material="bucket_mat"/>
              <geom name="bucket_edge" type="box" size="{bu*0.5} {bw*0.4} 0.025"
                    pos="{bu*0.48} 0 {-bu*0.4}" material="bucket_mat"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="main_cam" pos="-5 9 7" xyaxes="0.85 0 0 0 -0.35 0.9" fovy="55"/>
  </worldbody>

  <actuator>
    <position name="act_swing"  joint="swing"  kp="200" kv="20"/>
    <position name="act_boom"   joint="boom"   kp="300" kv="30"/>
    <position name="act_arm"    joint="arm"    kp="200" kv="20"/>
    <position name="act_bucket" joint="bucket" kp="150" kv="15"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# FK helper (for elevation map overlay)
# ---------------------------------------------------------------------------

def excavator_fk(params: dict, qpos: np.ndarray) -> dict:
    """World-frame positions for key body points."""
    swing, boom, arm, bucket = qpos
    bl, al, bu = params["boom_len"], params["arm_len"], params["bucket_len"]
    ch = params["cab_height"]
    bi = params.get("boom_init_angle", 1.0)
    ai = params.get("arm_init_angle", 1.5)

    base = np.array([0.0, 0.0, 0.0])
    boom_pivot = np.array([0.0, 0.0, 0.5 + ch * 0.9])

    # Boom tip in cab frame, then rotate by swing
    bm = boom + bi
    bx = bl * np.sin(bm); bz = bl * np.cos(bm)
    cs, sn = np.cos(swing), np.sin(swing)
    boom_tip = boom_pivot + np.array([bx * cs, bx * sn, bz])

    # Arm tip
    total = np.pi - bm - (arm + ai)
    ax = al * np.sin(total); az = -al * np.cos(total)
    arm_tip = boom_tip + np.array([ax * cs, ax * sn, az])

    # Bucket tip
    ba = total + bucket
    bkx = bu * np.sin(ba); bkz = -bu * np.cos(ba)
    bucket_tip = arm_tip + np.array([bkx * cs, bkx * sn, bkz])

    # Bucket tooth
    bex = (bu + 0.08) * np.sin(ba); bez = -(bu + 0.08) * np.cos(ba)
    bucket_edge = arm_tip + np.array([bex * cs, bex * sn, bez])

    return {"base": base, "boom_pivot": boom_pivot, "boom_tip": boom_tip,
            "arm_tip": arm_tip, "bucket_tip": bucket_tip, "bucket_edge": bucket_edge}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class ExcavatorSim:
    """MuJoCo physics + GPU OpenGL rendering."""

    def __init__(self, excavator_type: str = "75",
                 img_width: int = 640, img_height: int = 480,
                 elevation_res: int = 200, seed: Optional[int] = None):
        self.excavator_type = excavator_type
        self.params = EXCAVATOR_PARAMS[excavator_type]
        self.img_width = img_width
        self.img_height = img_height
        self.elevation_res = elevation_res
        self.mpp = self.params["meter_per_pixel"]
        self.extent = elevation_res * self.mpp
        self.rng = np.random.RandomState(seed)
        self._mounds = []
        self._renderer = None

        self._build_model()

    @property
    def elevation_extent(self) -> float:
        """Physical width of the square elevation map in metres.

        Kept as a named alias for callers that describe the extent as an
        elevation-map property; ``extent`` remains available for compatibility.
        """
        return self.extent

    def _build_model(self):
        """(Re)build MuJoCo model with fresh terrain."""
        self._mounds = self._make_mounds()
        xml = _build_mjcf(self.params, DEFAULT_JOINT_RANGE,
                          self._mounds, self.rng)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

        # (Re)create GPU renderer
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self._model, self.img_height, self.img_width)

        # Joint/actuator lookups
        self._jid = {n: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in ["swing", "boom", "arm", "bucket"]}
        self._aid = {n: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}")
                     for n in ["swing", "boom", "arm", "bucket"]}

    def _make_mounds(self):
        mounds = []
        for _ in range(self.rng.randint(3, 9)):
            mounds.append((self.rng.uniform(2, 12) * self.rng.choice([-1, 1]),
                           self.rng.uniform(3, 15),
                           self.rng.uniform(1.0, 4.0), self.rng.uniform(1.0, 4.0),
                           self.rng.uniform(0.3, 1.8)))
        return mounds

    def rebuild_terrain(self):
        """New terrain for a new episode."""
        self._build_model()

    def get_qpos(self) -> np.ndarray:
        return np.array([self._data.qpos[self._model.jnt_qposadr[self._jid[n]]]
                         for n in ["swing", "boom", "arm", "bucket"]], dtype=np.float32)

    def reset(self, qpos: Optional[np.ndarray] = None):
        mujoco.mj_resetData(self._model, self._data)
        self.set_qpos(qpos if qpos is not None
                      else np.array([0.0, 0.785, -1.047, 0.524], dtype=np.float64))

    def set_qpos(self, qpos: np.ndarray):
        for i, n in enumerate(["swing", "boom", "arm", "bucket"]):
            self._data.qpos[self._model.jnt_qposadr[self._jid[n]]] = float(qpos[i])
        mujoco.mj_forward(self._model, self._data)

    def step(self, action: np.ndarray, n_substeps: int = 5):
        for i, n in enumerate(["swing", "boom", "arm", "bucket"]):
            self._data.ctrl[self._aid[n]] = float(action[i])
        for _ in range(n_substeps):
            mujoco.mj_step(self._model, self._data)

    # ── GPU-accelerated RGB rendering ───────────────────────────────────

    def render_main(self) -> np.ndarray:
        """Render main RGB view via MuJoCo GPU renderer → [H, W, 3] uint8 BGR."""
        self._renderer.update_scene(self._data, camera="main_cam")
        rgb = self._renderer.render()              # [H, W, 3] uint8 RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── CPU elevation map ───────────────────────────────────────────────

    def compute_elevation(self) -> np.ndarray:
        """Compute top-down elevation map → [E, E, 3] uint8 BGR."""
        E, mpp = self.elevation_res, self.mpp
        elev = np.zeros((E, E), dtype=np.float32)

        # Mounds
        for (cx, cy, sx, sy, h) in self._mounds:
            pcx, pcy = int(cx / mpp + E / 2), int(cy / mpp + E / 2)
            prx, pry = max(1, int(sx / mpp)), max(1, int(sy / mpp))
            y, x = np.ogrid[:E, :E]
            d2 = ((x - pcx) / prx) ** 2 + ((y - pcy) / pry) ** 2
            mask = d2 <= 1
            if mask.any():
                elev = np.maximum(elev, h * np.maximum(0, 1 - d2) * mask)

        # Excavator body
        fk = excavator_fk(self.params, self.get_qpos())
        for key in ["base", "boom_pivot", "boom_tip", "arm_tip", "bucket_tip", "bucket_edge"]:
            pos = fk[key]
            px, py = int(pos[0] / mpp + E / 2), int(pos[1] / mpp + E / 2)
            r = max(2, int(0.15 / mpp))
            y, x = np.ogrid[:E, :E]
            mask = (x - px) ** 2 + (y - py) ** 2 <= r ** 2
            if mask.any():
                elev[mask] = np.maximum(elev[mask], float(pos[2]))

        # Colorize
        emin, emax = elev.min(), max(elev.max(), elev.min() + 0.01)
        en = np.clip((elev - emin) / (emax - emin), 0, 1)
        cmap = TERRAIN_COLORMAP
        nc = cmap.shape[0]
        idx_f = en * (nc - 1)
        lo = np.clip(np.floor(idx_f).astype(np.int32), 0, nc - 1)
        hi = np.clip(np.ceil(idx_f).astype(np.int32), 0, nc - 1)
        t = (idx_f - lo)[:, :, np.newaxis]
        colored = cmap[lo] * (1 - t) + cmap[hi] * t
        colored = np.clip(colored * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._model = None
        self._data = None


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_digging_trajectory(
    rng: np.random.RandomState, n_steps: int = 150,
    joint_range: np.ndarray = DEFAULT_JOINT_RANGE,
    speed_variation: float = 0.2) -> np.ndarray:
    """Scripted digging cycle → [n_steps, 4] float32."""
    jr = joint_range
    home = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.3, 0.8),
                     rng.uniform(-1.2, -0.5), rng.uniform(0.2, 0.8)])
    dig = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.3),
                    rng.uniform(-1.5, -0.8), rng.uniform(1.5, 2.2)])
    dump = np.array([rng.uniform(1.5, 3.0) * rng.choice([-1, 1]),
                     rng.uniform(0.5, 1.0), rng.uniform(-1.0, -0.4),
                     rng.uniform(-0.5, 0.3)])

    phases = [
        (home, home*0.7+dig*0.3, 0.15), (home*0.7+dig*0.3, dig, 0.15),
        (dig, dig+np.array([0,0,0.1,0.2]), 0.10),
        (dig+np.array([0,0,0.1,0.2]), home*0.5+dump*0.5, 0.15),
        (home*0.5+dump*0.5, dump, 0.15),
        (dump, dump*np.array([1,1,1,-0.2]), 0.10),
        (dump*np.array([1,1,1,-0.2]), home, 0.20),
    ]
    durs = [max(5, int(f * n_steps * (1 + rng.uniform(-speed_variation, speed_variation))))
            for _, _, f in phases]
    total = sum(durs)
    durs = [max(4, int(d * n_steps / total)) for d in durs]
    while sum(durs) < n_steps: durs[rng.randint(0, len(durs)-1)] += 1
    while sum(durs) > n_steps:
        i = rng.randint(0, len(durs)-1)
        if durs[i] > 4: durs[i] -= 1

    traj = []
    for (s, e, _), dur in zip(phases, durs):
        for t in range(dur):
            a = t / max(dur-1, 1); a = a**2 * (3 - 2*a)
            traj.append(s * (1-a) + e*a + rng.normal(0, 0.005, 4))
    traj = np.array(traj[:n_steps], dtype=np.float32)
    for j in range(4): traj[:, j] = np.clip(traj[:, j], jr[j, 0], jr[j, 1])
    return traj


def generate_idle_trajectory(rng: np.random.RandomState, n_steps: int = 100,
                              joint_range=DEFAULT_JOINT_RANGE) -> np.ndarray:
    jr = joint_range
    home = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.3, 0.8),
                     rng.uniform(-1.2, -0.5), rng.uniform(0.2, 0.8)])
    traj = [home]
    for _ in range(n_steps-1):
        traj.append(np.clip(traj[-1] + rng.normal(0, 0.02, 4), jr[:, 0], jr[:, 1]))
    return np.array(traj, dtype=np.float32)
