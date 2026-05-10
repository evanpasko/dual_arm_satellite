"""Matplotlib helpers for ``base_link`` pose time series."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from physics_sim.base_state import PoseTimeSeries


def plot_base_link_pose(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    quaternions_xyzw: np.ndarray,
    *,
    title: str = "base_link pose vs time",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Plot world-frame position (m) and orientation quaternion (x,y,z,w) vs time.

    Parameters
    ----------
    times_s
        Shape (N,).
    positions_m
        Shape (N, 3) — columns x, y, z.
    quaternions_xyzw
        Shape (N, 4) — columns x, y, z, w (ROS / ``geometry_msgs`` order).
    save_path
        If set, save figure to this path (extension sets format, e.g. ``.png``, ``.pdf``).
    show
        If True, call ``plt.show()`` (set False in headless CI).
    """
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    positions_m = np.asarray(positions_m, dtype=float).reshape(-1, 3)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=float).reshape(-1, 4)
    if times_s.size == 0:
        raise ValueError("No samples to plot")
    if positions_m.shape[0] != times_s.size or quaternions_xyzw.shape[0] != times_s.size:
        raise ValueError("times_s, positions_m, and quaternions_xyzw must have the same length")

    fig, (ax_pos, ax_quat) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), constrained_layout=True
    )
    fig.suptitle(title)

    ax_pos.plot(times_s, positions_m[:, 0], label="x")
    ax_pos.plot(times_s, positions_m[:, 1], label="y")
    ax_pos.plot(times_s, positions_m[:, 2], label="z")
    ax_pos.set_ylabel("position (m)")
    ax_pos.legend(loc="upper right")
    ax_pos.grid(True, alpha=0.3)

    ax_quat.plot(times_s, quaternions_xyzw[:, 0], label="qx")
    ax_quat.plot(times_s, quaternions_xyzw[:, 1], label="qy")
    ax_quat.plot(times_s, quaternions_xyzw[:, 2], label="qz")
    ax_quat.plot(times_s, quaternions_xyzw[:, 3], label="qw")
    ax_quat.set_xlabel("time (s)")
    ax_quat.set_ylabel("quaternion (xyzw)")
    ax_quat.legend(loc="upper right")
    ax_quat.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_pose_log(
    log: PoseTimeSeries,
    *,
    title: str = "base_link pose vs time",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """Convenience wrapper: :class:`PoseTimeSeries` → :func:`plot_base_link_pose`."""
    t, p, q = log.as_arrays()
    return plot_base_link_pose(
        t, p, q, title=title, save_path=save_path, show=show
    )
