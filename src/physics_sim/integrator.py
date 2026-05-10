"""Simple pose integration: constant acceleration in world (linear) and body (angular)."""

from __future__ import annotations

import numpy as np

from physics_sim.base_state import normalize_quaternion_xyzw


def _xyzw_to_hamilton(q_xyzw: np.ndarray) -> np.ndarray:
    """``[x,y,z,w]`` → ``[w,x,y,z]`` column for Ω algebra."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=float)


def _hamilton_to_xyzw(q_h: np.ndarray) -> np.ndarray:
    w, x, y, z = q_h
    return np.array([x, y, z, w], dtype=float)


def _omega_matrix_body(omega: np.ndarray) -> np.ndarray:
    """4×4 matrix Ω(ω) with ω = [ωx, ωy, ωz] in body frame (rad/s)."""
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
        dtype=float,
    )


def integrate_pose_constant_accel(
    *,
    duration_s: float,
    dt_s: float,
    position0_m: np.ndarray,
    quaternion_xyzw0: np.ndarray,
    velocity_world_m_s: np.ndarray,
    angular_velocity_body_rad_s: np.ndarray,
    acceleration_world_m_s2: np.ndarray,
    angular_accel_body_rad_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Semi-implicit Euler: world-frame ``p, v, a``; body-frame ``ω, α``.

    Quaternion kinematics use Hamilton ``[w,x,y,z]`` internally; I/O stays ``xyzw``
    (ROS / :class:`~physics_sim.base_state.BaseLinkPose`).

    Parameters
    ----------
    duration_s
        Simulation length (s). Samples include ``t=0`` and ``t=duration_s``.
    dt_s
        Nominal step; actual grid is ``np.linspace(0, duration_s, ceil(duration/dt)+1)``.
    """
    if duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    n = max(2, int(np.ceil(duration_s / dt_s)) + 1)
    times = np.linspace(0.0, float(duration_s), n)
    delta = float(times[1] - times[0])

    p = np.asarray(position0_m, dtype=float).reshape(3).copy()
    q = normalize_quaternion_xyzw(quaternion_xyzw0)
    v = np.asarray(velocity_world_m_s, dtype=float).reshape(3).copy()
    omega = np.asarray(angular_velocity_body_rad_s, dtype=float).reshape(3).copy()
    a_w = np.asarray(acceleration_world_m_s2, dtype=float).reshape(3).copy()
    alpha_b = np.asarray(angular_accel_body_rad_s2, dtype=float).reshape(3).copy()

    pos_out = np.zeros((n, 3), dtype=float)
    quat_out = np.zeros((n, 4), dtype=float)
    pos_out[0] = p
    quat_out[0] = q

    for i in range(1, n):
        omega = omega + alpha_b * delta
        v = v + a_w * delta
        p = p + v * delta
        q_h = _xyzw_to_hamilton(q)
        dq_h = 0.5 * _omega_matrix_body(omega) @ q_h
        q_h = q_h + dq_h * delta
        q = normalize_quaternion_xyzw(_hamilton_to_xyzw(q_h))

        pos_out[i] = p
        quat_out[i] = q

    return times, pos_out, quat_out
