"""3D visualization of thruster mounts and thrust directions in ``base_link`` frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from physics_sim.fk_chain import (
    T_base_thruster,
    arm_joint_slice,
    arm_link_frame_origins_base_m,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def thruster_origin_and_thrust_axis_base(
    arm,
    joint_positions_rad: np.ndarray,
    *,
    thrust_axis_thruster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thruster frame origin and unit thrust direction, both expressed in **base_link** frame.

    Thrust is along **+Y** of the thruster link (same convention as :class:`~physics_sim.thruster.Thruster`).
    """
    q = np.asarray(joint_positions_rad, dtype=float).reshape(4)
    T = T_base_thruster(arm, q)
    origin_b = T[:3, 3].copy()
    R = T[:3, :3]
    if thrust_axis_thruster is None:
        axis_t = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        axis_t = np.asarray(thrust_axis_thruster, dtype=float).reshape(3)
    n = np.linalg.norm(axis_t)
    if n < 1e-15:
        raise ValueError("Thrust axis has zero length")
    axis_t = axis_t / n
    dir_b = R @ axis_t
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-15)
    return origin_b, dir_b


def _set_axes_equal_3d(ax: "Axes") -> None:
    """Roughly equal scale on x, y, z for an Axes3D."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    r = 0.5 * float(np.max(spans)) + 1e-6
    ax.set_xlim3d(centers[0] - r, centers[0] + r)
    ax.set_ylim3d(centers[1] - r, centers[1] + r)
    ax.set_zlim3d(centers[2] - r, centers[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def redraw_thrusters_in_base_frame(
    ax3d: "Axes",
    engine,
    *,
    arrow_length_m: float = 0.22,
    base_axes_length_m: float = 0.12,
) -> None:
    """
    Clear ``ax3d`` and draw **base_link** XYZ axes, arm skeleton (link frames 1–4), and
    left/right thruster origins with **+Y** thrust arrows.

    ``engine`` must provide ``left_arm``, ``right_arm``, and ``joint_positions_rad``.
    """
    ax3d.cla()
    ax3d.set_title("Thrusters & arms in base_link frame")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")

    L = float(base_axes_length_m)
    ax3d.plot([0.0, L], [0.0, 0.0], [0.0, 0.0], color="#b91c1c", linewidth=2.0, label="base +x")
    ax3d.plot([0.0, 0.0], [0.0, L], [0.0, 0.0], color="#15803d", linewidth=2.0, label="base +y")
    ax3d.plot([0.0, 0.0], [0.0, 0.0], [0.0, L], color="#1d4ed8", linewidth=2.0, label="base +z")
    ax3d.plot([0.0], [0.0], [0.0], "k.", markersize=10, label="base origin")

    for side, color, skel in (
        ("left", "#d97706", "#92400e"),
        ("right", "#16a34a", "#14532d"),
    ):
        arm = engine.left_arm if side == "left" else engine.right_arm
        q = arm_joint_slice(engine.joint_positions_rad, side)  # type: ignore[arg-type]
        origins = arm_link_frame_origins_base_m(arm, q)
        ax3d.plot(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            "-",
            color=skel,
            linewidth=2.2,
            label=f"{side} links 1–4 (frame chain)",
        )
        o, d = thruster_origin_and_thrust_axis_base(arm, q)
        ax3d.plot(
            [o[0]],
            [o[1]],
            [o[2]],
            "o",
            color=color,
            markersize=8,
            label=f"{side} thruster (+Y bore)",
        )
        ax3d.quiver(
            float(o[0]),
            float(o[1]),
            float(o[2]),
            float(d[0]),
            float(d[1]),
            float(d[2]),
            length=float(arrow_length_m),
            normalize=True,
            color=color,
            linewidth=1.5,
        )

    ax3d.legend(loc="upper left", fontsize=7)
    _set_axes_equal_3d(ax3d)
