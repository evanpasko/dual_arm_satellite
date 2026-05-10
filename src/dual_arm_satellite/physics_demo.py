"""
Run a constant-acceleration kinematic trajectory for ``base_link`` and plot pose vs time.

Linear motion is integrated in the **world** frame; angular velocity and angular acceleration
are in the **body** frame (roll / pitch / yaw rates about body x, y, z).

Example::

    uv run dual-arm-satellite-sim --vx 0.01 --wr 0.02 --ax 0.001 --alphar 0.0001
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from physics_sim import BaseLinkPose, PhysicsSimEngine, plot_base_link_pose
from physics_sim.integrator import integrate_pose_constant_accel


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate base_link pose under constant world linear and body angular "
        "acceleration; plot position (m) and quaternion (xyzw) vs time."
    )
    p.add_argument("--vx", type=float, default=0.0, help="initial world v_x (m/s)")
    p.add_argument("--vy", type=float, default=0.0, help="initial world v_y (m/s)")
    p.add_argument("--vz", type=float, default=0.0, help="initial world v_z (m/s)")
    p.add_argument(
        "--wr",
        type=float,
        default=0.0,
        help="initial body ω_roll about body x (rad/s)",
    )
    p.add_argument(
        "--wp",
        type=float,
        default=0.0,
        help="initial body ω_pitch about body y (rad/s)",
    )
    p.add_argument(
        "--wy",
        type=float,
        default=0.0,
        help="initial body ω_yaw about body z (rad/s)",
    )
    p.add_argument("--ax", type=float, default=0.0, help="world a_x (m/s²)")
    p.add_argument("--ay", type=float, default=0.0, help="world a_y (m/s²)")
    p.add_argument("--az", type=float, default=0.0, help="world a_z (m/s²)")
    p.add_argument(
        "--alphar",
        type=float,
        default=0.0,
        help="body angular accel about x (rad/s²)",
    )
    p.add_argument(
        "--alphap",
        type=float,
        default=0.0,
        help="body angular accel about y (rad/s²)",
    )
    p.add_argument(
        "--alphay",
        type=float,
        default=0.0,
        help="body angular accel about z (rad/s²)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="simulation duration in seconds (default: 60)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="internal timestep (~1/sample density); default 0.02 s",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional path to save figure (e.g. pose.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive plot window",
    )
    return p.parse_args(argv)


def run_demo(
    *,
    velocity_world: np.ndarray,
    omega_body: np.ndarray,
    accel_world: np.ndarray,
    alpha_body: np.ndarray,
    duration_s: float = 60.0,
    dt_s: float = 0.02,
    initial_pose: Optional[BaseLinkPose] = None,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate pose, push results through :class:`PhysicsSimEngine` logging, and plot.

    Returns ``(times_s, positions_m, quaternions_xyzw)``.
    """
    engine = PhysicsSimEngine.from_urdf(initial_base_pose=initial_pose)
    pose0 = engine.base_pose

    times, pos, quat = integrate_pose_constant_accel(
        duration_s=duration_s,
        dt_s=dt_s,
        position0_m=pose0.position_m,
        quaternion_xyzw0=pose0.quaternion_xyzw,
        velocity_world_m_s=velocity_world,
        angular_velocity_body_rad_s=omega_body,
        acceleration_world_m_s2=accel_world,
        angular_accel_body_rad_s2=alpha_body,
    )

    engine.pose_log.clear()
    for i in range(len(times)):
        engine.set_base_pose(BaseLinkPose(pos[i], quat[i]))
        engine.record_base_pose(float(times[i]))

    plot_base_link_pose(
        times,
        pos,
        quat,
        title="base_link pose (constant world linear & body angular accel)",
        save_path=save_path,
        show=show_plot,
    )
    return times, pos, quat


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    v = np.array([args.vx, args.vy, args.vz], dtype=float)
    w = np.array([args.wr, args.wp, args.wy], dtype=float)
    a = np.array([args.ax, args.ay, args.az], dtype=float)
    alpha = np.array([args.alphar, args.alphap, args.alphay], dtype=float)

    run_demo(
        velocity_world=v,
        omega_body=w,
        accel_world=a,
        alpha_body=alpha,
        duration_s=args.duration,
        dt_s=args.dt,
        save_path=args.save,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
