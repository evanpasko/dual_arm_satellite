"""Inverse kinematics for one 4-DoF arm in the ``base_link`` frame.

The solver is a damped-least-squares (Levenberg-Marquardt-style) iteration on the joint
vector ``q`` (4 angles). It supports three task modes:

* **Position only**: drive the thruster origin to a target point in ``base_link`` frame.
* **Direction only**: drive the thruster bore (default ``+Y`` of the thruster link)
  to a target unit vector in ``base_link`` frame.
* **Position + direction**: weighted combination of both.

This solves a single arm independently; for the dual-arm controller, call it per side
and combine results upstream. Working in ``base_link`` frame means the IK does not need
the satellite's world pose - thrust direction / application targets coming from a
body-frame controller drop straight in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from robot_description.fk import T_base_thruster
from robot_description.models import RobotArmDefinition

DEFAULT_THRUST_AXIS_THRUSTER = np.array([0.0, 1.0, 0.0], dtype=float)


@dataclass
class IKResult:
    """Outcome of a single-arm IK solve."""

    joint_positions_rad: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    direction_error_rad: float
    final_cost: float


def thruster_pose_in_base(
    arm: RobotArmDefinition,
    joint_positions_rad: np.ndarray,
    *,
    thrust_axis_thruster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thruster-frame origin and unit thrust direction, both in ``base_link`` frame.

    Returns ``(p_base_m, d_base_unit)``. The thrust direction defaults to ``+Y`` of the
    thruster link (matches :class:`physics_sim.thruster.Thruster`).
    """
    q = np.asarray(joint_positions_rad, dtype=float).reshape(4)
    T = T_base_thruster(arm, q)
    p = T[:3, 3].copy()
    axis_t = (
        DEFAULT_THRUST_AXIS_THRUSTER
        if thrust_axis_thruster is None
        else np.asarray(thrust_axis_thruster, dtype=float).reshape(3)
    )
    n = float(np.linalg.norm(axis_t))
    if n < 1e-15:
        raise ValueError("thrust_axis_thruster has zero length")
    d = T[:3, :3] @ (axis_t / n)
    d = d / (float(np.linalg.norm(d)) + 1e-15)
    return p, d


def _clip_to_limits(q: np.ndarray, arm: RobotArmDefinition) -> np.ndarray:
    out = q.copy()
    for i, j in enumerate(arm.joints):
        out[i] = float(np.clip(out[i], j.limit_lower_rad, j.limit_upper_rad))
    return out


def _build_residual(
    arm: RobotArmDefinition,
    target_position_base_m: Optional[np.ndarray],
    target_thrust_direction_base: Optional[np.ndarray],
    position_weight: float,
    direction_weight: float,
    thrust_axis_thruster: Optional[np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    sqrt_wp = math.sqrt(position_weight) if target_position_base_m is not None else 0.0
    sqrt_wd = math.sqrt(direction_weight) if target_thrust_direction_base is not None else 0.0

    def residual(q: np.ndarray) -> np.ndarray:
        p, d = thruster_pose_in_base(arm, q, thrust_axis_thruster=thrust_axis_thruster)
        parts = []
        if target_position_base_m is not None:
            parts.append(sqrt_wp * (p - target_position_base_m))
        if target_thrust_direction_base is not None:
            parts.append(sqrt_wd * (d - target_thrust_direction_base))
        return np.concatenate(parts)

    return residual


def _numerical_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    r0 = fn(q)
    n_q = q.shape[0]
    J = np.zeros((r0.shape[0], n_q), dtype=float)
    for i in range(n_q):
        q_p = q.copy()
        q_p[i] += eps
        r_p = fn(q_p)
        J[:, i] = (r_p - r0) / eps
    return J


def solve_arm_ik(
    arm: RobotArmDefinition,
    *,
    target_position_base_m: Optional[np.ndarray] = None,
    target_thrust_direction_base: Optional[np.ndarray] = None,
    initial_joint_positions_rad: Optional[np.ndarray] = None,
    position_weight: float = 1.0,
    direction_weight: float = 1.0,
    max_iterations: int = 200,
    position_tolerance_m: float = 1e-4,
    direction_tolerance_rad: float = 1e-4,
    damping: float = 1e-2,
    thrust_axis_thruster: Optional[np.ndarray] = None,
    enforce_joint_limits: bool = True,
) -> IKResult:
    """Damped least-squares IK for a single 4-DoF arm, posed in ``base_link`` frame.

    At least one of ``target_position_base_m`` and ``target_thrust_direction_base`` must
    be set. ``position_weight``/``direction_weight`` trade the two task residuals when
    both are active (each is scaled by ``sqrt(weight)`` inside the residual). The 4
    joints are clipped to URDF limits each iteration when ``enforce_joint_limits`` is
    true.

    Direction error is reported as the angle (rad) between current and target unit
    vectors. Position error is the Euclidean distance (m). Convergence requires whichever
    targets are active to all be within their tolerance.
    """
    if target_position_base_m is None and target_thrust_direction_base is None:
        raise ValueError(
            "Need at least one of target_position_base_m or target_thrust_direction_base"
        )

    p_target = (
        np.asarray(target_position_base_m, dtype=float).reshape(3)
        if target_position_base_m is not None
        else None
    )
    d_target: Optional[np.ndarray] = None
    if target_thrust_direction_base is not None:
        d_raw = np.asarray(target_thrust_direction_base, dtype=float).reshape(3)
        n_d = float(np.linalg.norm(d_raw))
        if n_d < 1e-12:
            raise ValueError("target_thrust_direction_base has zero length")
        d_target = d_raw / n_d

    q = (
        np.asarray(initial_joint_positions_rad, dtype=float).reshape(4).copy()
        if initial_joint_positions_rad is not None
        else np.zeros(4, dtype=float)
    )
    if enforce_joint_limits:
        q = _clip_to_limits(q, arm)

    residual = _build_residual(
        arm,
        p_target,
        d_target,
        position_weight,
        direction_weight,
        thrust_axis_thruster,
    )

    eye4 = np.eye(4, dtype=float)
    lam_sq = float(damping) ** 2
    converged = False
    iterations = 0
    pos_err = 0.0
    dir_err = 0.0
    cost = math.inf

    for k in range(int(max_iterations)):
        iterations = k + 1
        r = residual(q)
        cost = float(r @ r)

        p_now, d_now = thruster_pose_in_base(
            arm, q, thrust_axis_thruster=thrust_axis_thruster
        )
        pos_err = float(np.linalg.norm(p_now - p_target)) if p_target is not None else 0.0
        if d_target is not None:
            cos_theta = float(np.clip(d_now @ d_target, -1.0, 1.0))
            dir_err = math.acos(cos_theta)
        else:
            dir_err = 0.0

        pos_ok = p_target is None or pos_err < position_tolerance_m
        dir_ok = d_target is None or dir_err < direction_tolerance_rad
        if pos_ok and dir_ok:
            converged = True
            break

        J = _numerical_jacobian(residual, q)
        H = J.T @ J + lam_sq * eye4
        g = J.T @ r
        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        q_new = q - dq
        if enforce_joint_limits:
            q_new = _clip_to_limits(q_new, arm)
        q = q_new

    return IKResult(
        joint_positions_rad=q,
        converged=converged,
        iterations=iterations,
        position_error_m=pos_err,
        direction_error_rad=dir_err,
        final_cost=cost,
    )
