# Isaac Sim Static-Terrain Excavator Data Design

## Goal

Create a new Isaac Sim pipeline that generates geometrically consistent, custom
excavator VLA trajectories. It must produce RGB, metric height maps, depth,
joint state, and next-step actions without requiring 75/490 CAD assets or
modifying the existing MuJoCo pipeline.

## Scope

The first version uses static procedurally generated terrain within an episode.
The excavator moves over a fixed terrain; bucket-driven soil deformation is out
of scope. Terrain, excavator dimensions, materials, lighting, and camera
parameters vary between episodes.

## Scene and motion

Build a custom articulated excavator from parameterized USD primitives: tracked
base, upper carriage, cab, boom, arm, and bucket. Four named joints represent
swing, boom, arm, and bucket. A scripted position-target trajectory supplies
kinematically valid motion while the simulator provides collision-aware scene
state and rendering.

## Sensors and height map

Use a perspective main camera with a look-at target in the excavator work area,
so the excavator stays in frame. Use a calibrated top-down orthographic depth
camera over a fixed square work area. Convert its depth result into a metric
world-Z raster with resolution 200 by 200; this raster is the authoritative
height map. Create a colorized BGR elevation image from the same metric raster
for compatibility and visualization.

## Data contract

Each HDF5 episode preserves the existing required datasets:

- `observations/images/main`: BGR uint8 RGB frames
- `observations/images/elevation`: BGR uint8 colorized height-map frames
- `observations/qpos`: float32 `[N, 4]`
- `action`: float32 next-step qpos `[N, 4]`

Additional Isaac-only information is stored under `observations/sim/`: metric
height maps, depth images, camera intrinsics, camera extrinsics, and episode
metadata. Existing `ExcavatorDataset` can read the required datasets unchanged.

## Output and validation

The generator writes one HDF5 file per episode and a preview tool renders a
side-by-side MP4 containing RGB, colorized height map, and joint traces. Before
writing an episode, validation checks that the RGB camera contains excavator
pixels, the metric height map has finite non-flat values, and its dimensions
match the elevation image.

## Delivery

New scripts live under `scripts/isaac_sim/`; MuJoCo files remain intact. Isaac
Sim runs on the server environment. Changes are pushed to the existing GitHub
remote; the server receives them through `git pull origin main`.
