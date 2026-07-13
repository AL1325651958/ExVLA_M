# V11–V12 Causal Visual-Dynamics Design

## Causal formulation

The physical process is joint state and control causing excavator geometry and
therefore image formation. The VLA model is an inverse observer: it uses a
history of RGB and elevation observations to estimate the current latent motion
state and predict the next joint state. Motion images never act as a cause of
the joints.

## V11: temporal observation evidence

V11 keeps pure-visual inference. Its inputs are RGB and elevation image
histories plus their adjacent-frame residuals. Residuals are observation
features that expose already-observed geometry changes; they are not actions,
targets, or control inputs. A lightweight residual encoder fuses them with the
existing visual tokens before the temporal mask and action heads predict
`qpos[t+1]`.

V11 also corrects the scheduler unit mismatch so scheduler stepping and warmup
are both batch-based, adds early stopping on validation loss, and logs metrics
by joint, excavator type, and episode. It continues to return the V9/V10
three-tensor inference API.

Clip augmentation is temporally coherent: a brightness/contrast sample is
shared by RGB and elevation frames throughout one sequence. Horizontal flipping
is deliberately disabled because it would alter excavator handedness and joint
semantics. `visualize_yolo.py` detects V11 from the `motion_adapter` checkpoint
keys, builds the V11 model, and defaults to the raw last-frame masks. V9/V10
checkpoints remain supported.

## V12: optical-flow visual state estimation

V12 is a future design only; it is not implemented by V11. It replaces or augments frame residuals with precomputed, cached dense optical
flow. Flow is calculated offline from adjacent RGB and elevation frames, then
read as a visual observation at training and inference. It encodes apparent
pixel displacement, not a control signal.

V12 adds a training-only latent state-transition auxiliary objective. A visual
latent is supervised to estimate the current joint state, while a small
transition head is supervised to map that training-only state to the next joint
state. Neither qpos nor the transition head is required by the deployed visual
forward path.

## Evaluation

Compare V10, V11, and V12 with the identical stratified split. Primary metrics
are per-joint MAE and R², with Swing reported separately by excavator type and
episode. Select checkpoints by mean validation MAE; retain an early-stopping
patience of five validation epochs.

## Compatibility

V11 and V12 use separate training entry points and checkpoint version metadata.
The visualizer continues to load V9/V10 checkpoints. V11/V12 inference accepts
only image histories and excavator ID; it ignores qpos if retained in a
compatibility signature.
