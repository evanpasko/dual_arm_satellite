"""Quaternion kinematics: body-frame angular velocity (rad/s), ``(x,y,z,w)`` order."""

from __future__ import annotations

import numpy as np

from physics_sim.base_state import normalize_quaternion_xyzw


def quat_derivative_xyzw_body_rates(
    q_xyzw: np.ndarray, omega_body_rad_s: np.ndarray
) -> np.ndarray:
    """
    Time derivative of unit quaternion ``q = [x,y,z,w]`` for body rates ``ω`` in the body frame.

    Convention matches :class:`~physics_sim.base_state.BaseLinkPose`: rotation **body → world**.
    """
    x, y, z, w = [float(v) for v in np.asarray(q_xyzw, dtype=float).reshape(4)]
    wx, wy, wz = [float(v) for v in np.asarray(omega_body_rad_s, dtype=float).reshape(3)]
    dq = np.zeros(4, dtype=float)
    dq[0] = 0.5 * (wx * w + wy * z - wz * y)
    dq[1] = 0.5 * (wy * w - wx * z + wz * x)
    dq[2] = 0.5 * (wz * w + wx * y - wy * x)
    dq[3] = 0.5 * (-wx * x - wy * y - wz * z)
    return dq


def integrate_quaternion_semi_implicit(
    q_xyzw: np.ndarray, omega_body_rad_s: np.ndarray, dt_s: float
) -> np.ndarray:
    """One semi-implicit Euler step; returns a normalized quaternion."""
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    w = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    dt = float(dt_s)
    dq = quat_derivative_xyzw_body_rates(q, w)
    return normalize_quaternion_xyzw(q + dq * dt)
