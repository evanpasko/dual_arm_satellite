# Dual Arm Satellite Simulation

A physics simulation and MVP control stack for a free-floating satellite whose only actuators are two 4-DoF robot arms, each tipped with a linear thruster. The project provides a rigid-body simulator, a URDF-driven robot description with forward / inverse kinematics, three runnable CLIs (a kinematic trajectory demo, an interactive keyboard teleop, and a closed-loop bang-bang position-control demo), and a pytest suite.

## Setup

It is recommended to use `uv` to manage python dependencies for this project. If you do not already have it installed, visit the [Astral UV site](https://docs.astral.sh/uv/getting-started/installation/) and run the installation command for your OS.

After cloning the repository, run a `uv sync` in the root directory of the project. Add `--group dev` if you also want to run the test suite.

```bash
git clone git@github.com:evanpasko/dual_arm_satellite.git
cd dual_arm_satellite
uv sync                 # runtime only
uv sync --group dev     # runtime + pytest
source .venv/bin/activate
```

## Project layout

Sources live under `src/`, organized into four installable packages:

- **`robot_description/`** — Static robot data, no simulation state.
  - `urdf/dual_arm_satellite.urdf` — Bus + two 4-DoF arms with `+Y`-bore thrusters at link 4.
  - `models.py`, `urdf_loading.py` — Parameter dataclasses and parser (`parse_robot_urdf`, `default_urdf_path`).
  - `fk.py` — Forward-kinematics primitives (`T_base_thruster`, `T_world_thruster`, `compute_thrust_axis_world`, `arm_link_frame_origins_base_m`, …).
  - `ik.py` — Per-arm damped-least-squares direction-only IK (`solve_arm_ik` → `IKResult`).
- **`physics_sim/`** — Dynamic simulation state (depends on `robot_description` for static data only).
  - `base_state.py`, `engine.py` — `BaseLinkPose`, `PoseTimeSeries`, `PhysicsSimEngine` (`from_urdf`, `step`, `apply_impulse_world`).
  - `integrator.py`, `attitude.py` — Kinematic trajectory integration and semi-implicit quaternion update.
  - `thruster.py` — `Thruster.fire(engine, throttle, impulse_window_s)`: linear impulse via FK + angular impulse `r × J` about the bus CoM.
  - `plotting.py`, `thrusters_3d.py`, `inertia_utils.py` — Matplotlib helpers and bus inertia utilities.
- **`controller/`** — Closed-loop pose controllers.
  - `types.py` — `StateError`, `ControlCommand` value-types (immutable, shape-checked).
  - `base.py` — `ControllerBaseClass` with `state_error`, `thruster_IK_simple` (delegates to `solve_arm_ik` per arm, returns the best-effort 8-vector), and an abstract `calculate_control`.
  - `bang_bang.py` — `BangBangController`: position-deadband MVP that points both thrusters at the target and fires at full throttle outside the deadband.
- **`dual_arm_satellite/`** — CLI entry points only.
  - `physics_demo.py` → `dual-arm-satellite-sim`
  - `teleop_live.py` → `dual-arm-satellite-teleop`
  - `basic_control_sim.py` → `dual-arm-satellite-control`

## Simplifying Assumptions

In order to achieve the solution in the requested time limit, the following simplifying assumptions were made to the model with brief justifications provided for each: 

### Massless Robot Arms and Effectors

Treating the arms as massless gives a much simpler approach to handling the dynamics of moving joints

### Joints Teleport - no internal torques/position control response

The joints of each robot arm will simply teleport to their requested position (either from teleop or target-pose controller). This removes the work of having to create an internal control/interpolation loop for each joint.

### Instantaneous Thrust

Thruster burns are modeled as short linear impulses along the thruster **+Y** axis through the URDF thruster frame origin. Each impulse updates linear momentum at the bus CoM and angular momentum about the CoM via **r × J** (lever arm **r** from the CoM to that point; **base_link** inertial origin is the CoM in the URDF). Between burns, orientation advances in the sim with body-frame angular velocity (no gravity).

### Satellite is not in Orbit

Per the problem description (pdf in base directory) the satellite is in 0 gravity so there is no orbital mechanics to include in the physics engine

## Teleop simulation

Interactive teleop with **two live matplotlib windows** in parallel: (1) `base_link` position (m) and quaternion (xyzw) vs time, and (2) a **3D base_link-frame** view with **RGB XYZ axes** at the base origin, **line segments along each arm** through link 4 (frame origins after joints 1–4), both thruster origins, and their **+Y thrust** directions (updated every animation frame as joints move).

**Where keys are read**

- **macOS / Linux, normal terminal:** stdin is switched to **raw mode** while the program runs. Type in **this same terminal**; keys are **not** fed to the shell line editor—they update the sim. Close the plot window to quit and restore the terminal.
- **Windows:** keys are read from the **console** via a background thread (type in the terminal window).
- **Non-TTY stdin** (e.g. piped input): keys must be sent to the **matplotlib figure** (click the plot so it has focus).

```bash
uv sync
uv run dual-arm-satellite-teleop
```

### Default key bindings

| Keys | Action |
|------|--------|
| `q` `w` `e` `r` `t` `y` `u` `i` | Positive increment on joints 1–8 (rad; default step ≈ 2°, see `--joint-step`) |
| `a` `s` `d` `f` `g` `h` `j` `k` | Negative increment on joints 1–8 |
| `[` | Single left-thruster impulse (this timestep window; see `--dt`) |
| `]` | Single right-thruster impulse |
| `0` | Reset simulation (pose, joints, velocity, plot history) |
| Close the figure | Quit |

Joints are **clamped** to URDF limits. Thruster impulse magnitude is `throttle × max_thrust × dt`; use `--thruster-throttle` for 0–1 scaling.

Useful flags: `--joint-step`, `--dt`, `--interval-ms`, `--left-max-thrust-n`, `--right-max-thrust-n`, `--max-plot-points` (rolling buffer), `--thruster-arrow-m` (thrust arrow length in the 3D view, meters), `--base-axes-m` (length of the red/green/blue +x/+y/+z segments from the base origin).

After you close the plot window, save the recorded pose time series to a file:

```bash
uv run dual-arm-satellite-teleop --save teleop_pose.png
```

## Bang-bang position-control simulation

Closed-loop demo where a `BangBangController` drives the bus toward a CLI-supplied world-frame target position. Shape mirrors the teleop sim — two live matplotlib windows (pose-vs-time with **dashed target lines**, plus a **fixed-cube 3D base-frame view** of the arms and thrust directions) — but the input each tick comes from the controller, not from the keyboard:

1. `BangBangController.calculate_control(engine, target_pose)` produces a `ControlCommand` (8 joint targets + per-side throttle).
2. Joints teleport to the commanded angles, then any commanded thruster fires for the current timestep.
3. `engine.step(dt)` advances the base pose under the resulting impulse.

The MVP is **position-only**: no velocity damping, so the bus reaches the target and then chatters around it in a bang-bang limit cycle — exactly the behaviour the simplification predicts.

```bash
# Default thrust (5 N per arm); close the window to quit.
uv run dual-arm-satellite-control --target 0.5 0.0 0.0

# Auto-stop after 10 s and save a pose plot, no interactive window.
uv run dual-arm-satellite-control --target 0.5 0.0 0.0 \
    --duration 10 --no-show --save bang_bang.png
```

Useful flags: `--target X Y Z` (required), `--max-thrust-n` (per-thruster peak force, N; default 5), `--dist-err-threshold-m` (position deadband; default 0.01 m), `--dt`, `--interval-ms`, `--duration` (auto-stop time; required with `--no-show`), `--view-half-m` (half-extent of the fixed 3D view cube, m; default 1.2), `--thruster-arrow-m`, `--base-axes-m`, `--max-plot-points`, `--save`, `--no-show`.

## Developer testing

Tests live under `tests/` and use [pytest](https://pytest.org/). The project configures `pythonpath = ["src"]` in `pyproject.toml` so imports such as `physics_sim` resolve without installing the package first.

Install the dev dependency group (includes pytest), then run the suite from the repository root:

```bash
uv sync --group dev
uv run --group dev pytest
```

### Test modules

- **`tests/test_physics_engine_urdf_defaults.py`** — Loads `PhysicsSimEngine.from_urdf()` with the default URDF, **prints** satellite mass, inertia, bus geometry, joint axes, link cylinder sizes, and thruster mounts, then asserts those values match the current `src/robot_description/urdf/dual_arm_satellite.urdf`. To see the printed report in the terminal, disable output capture:

  ```bash
  uv run --group dev pytest tests/test_physics_engine_urdf_defaults.py -s
  ```

- **`tests/test_base_pose_and_plotting.py`** — Covers optional `BaseLinkPose` / joint start state on the engine, `PoseTimeSeries` logging, and matplotlib helpers (`plot_base_link_pose`, `plot_pose_log`) saving PNGs under a temporary directory (non-interactive `Agg` backend).

- **`tests/test_integrator.py`** — Checks `integrate_pose_constant_accel`: quaternion norms stay unit length, trajectory length matches the time grid, and a pure translation case (no spin) leaves orientation near identity.

- **`tests/test_thruster.py`** — Verifies FK thrust direction (+Y thruster frame → world), unit norm, and that `Thruster.fire` applies the expected linear impulse at the bus CoM (`Δv = J / m`).

- **`tests/test_engine_step.py`** — Checks `PhysicsSimEngine.step`: `base_link` position advances by `v * dt` with fixed orientation.

- **`tests/test_attitude.py`** — Quaternion integration with zero body rate stays at identity.

- **`tests/test_thrusters_3d.py`** — Thruster origin and thrust axis in `base_link` frame from FK; `arm_link_frame_origins_base_m` shape `(5, 3)` for the arm skeleton.

- **`tests/test_ik.py`** — `robot_description.ik.solve_arm_ik` direction-only IK: convergence to seeded directions, handling of zero / near-singular targets, and joint-limit clamping behaviour.

- **`tests/test_controller_types.py`** — `StateError` and `ControlCommand` value-types: array shape validation, immutability, `.hold()` / `.zero()` factories, `is_*_firing` flags, joint-slice properties.

- **`tests/test_controller_base.py`** — `ControllerBaseClass.state_error` (position / rotation / velocity error correctness against known poses) and `thruster_IK_simple` (8-vector concatenation, both arms point along the requested body-frame direction).

- **`tests/test_bang_bang.py`** — `BangBangController.calculate_control` behaviour: deadband produces a hold command, outside the deadband both thrusters fire at full throttle along the target direction, returned joint array has the correct shape.

- **`tests/test_basic_control_sim.py`** — Headless smoke tests for `dual-arm-satellite-control`: the controller actually moves the bus toward the target, the `--save` path writes a plot file, and the `--no-show` CLI guard refuses to run without `--duration`.

### Optional CLI smoke test

The kinematic demo script is not part of pytest, but you can run it after a sync:

```bash
# No plotting
uv run dual-arm-satellite-sim --no-show --duration 1.0 --dt 0.05

# Plotting with optional velocity arguments
uv run dual-arm-satellite-sim --vx 1.0 --wy 1.0 --duration 1.0 --dt 0.05
```

Use `--save pose.png` to write the figure without opening a window.
