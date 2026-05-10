# Dual Arm Satellite Simulation

A physics simulation and MVP control solution for a free-floating satellite with two 4 DoF robot arms using thruster end-effectors.

## Simplifying Assumptions

In order to achieve the solution in the requested time limit, the following simplifying assumptions were made to the model with brief justifications provided for each: 

### Massless Robot Arms and Effectors

Treating the arms as massless gives a much simpler approach to handling the dynamics of moving joints

### Joints Teleport - no internal torques/position control response

The joints of each robot arm will simply teleport to their requested position (either from teleop or target-pose controller). This removes the work of having to create an internal control/interpolation loop for each joint.

### Instantaneous Thrust

Assuming zero-time linear impulses of the thruster end-effector rather than a more realistic change in momentum over time will help simplify the dynamics a lot when calculating the resulting linear and angular momentum transfer from a burn. 

### Satellite is not in Orbit

Per the problem description (pdf in base directory) the satellite is in 0 gravity so there is no orbital mechanics to include in the physics engine

### No Nearby Bodies

Also assume no nearby small bodies that could induce microgravity

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

### Optional CLI smoke test

The kinematic demo script is not part of pytest, but you can run it after a sync:

```bash
# No plotting
uv run dual-arm-satellite-sim --no-show --duration 1.0 --dt 0.05

# Plotting with optional velocity arguments
uv run dual-arm-satellite-sim --vx 1.0 --wy 1.0 --duration 1.0 --dt 0.05
```

Use `--save pose.png` to write the figure without opening a window.

## Teleop simulation

Interactive teleop with a **live matplotlib plot** of `base_link` position (m) and quaternion (xyzw) vs time.

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

Useful flags: `--joint-step`, `--dt`, `--interval-ms`, `--left-max-thrust-n`, `--right-max-thrust-n`, `--max-plot-points` (rolling buffer).

After you close the plot window, save the recorded pose time series to a file:

```bash
uv run dual-arm-satellite-teleop --save teleop_pose.png
```


