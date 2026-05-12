"""Abstract base class for satellite pose controllers.

Subclasses implement :meth:`ControllerBaseClass.calculate_control` to map the
current sim state + a target pose onto a :class:`ControlCommand`. The shared
helpers (:meth:`state_error`, :meth:`thruster_IK_simple`, :meth:`thruster_IK`)
are policy-independent geometry meant to be reused across controller flavours
(bang-bang today; PID / LQR later).

All public methods return one of the value-types defined in :mod:`controller.types`
so any subclass plugs into the same sim driver without bespoke I/O contracts.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from controller.types import ControlCommand, StateError
from physics_sim import BaseLinkPose, PhysicsSimEngine
from robot_description.fk import rotation_body_to_world_from_quat_xyzw
from robot_description.ik import solve_arm_ik


def _rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (theta * axis) of a 3x3 rotation matrix.

    Robust for ``0 <= theta < pi - eps``; degrades near ``theta == pi`` where the
    standard ``1 / sin(theta)`` formula loses precision. For an MVP pose
    controller (small attitude error per tick) the small-angle and mid-range
    cases are what we exercise.
    """
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-9:
        # theta ~ pi: use diagonal-based axis recovery, sign-corrected from skew part.
        diag = np.maximum((np.diag(R) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diag)
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3, dtype=float)
        return theta * (axis / n)
    factor = theta / (2.0 * sin_theta)
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )


class ControllerBaseClass:
    """Common interface for satellite pose controllers."""

    def __init__(self) -> None:
        pass

    def calculate_control(
        self,
        engine: PhysicsSimEngine,
        target_pose: BaseLinkPose,
    ) -> ControlCommand:
        """Map current sim state + target onto an actuation request.

        Subclasses implement their own policy (deadband + on/off for bang-bang,
        gain x error for PID, etc.). The return type is fixed
        (:class:`ControlCommand`) so any subclass plugs into the same sim
        driver.
        """
        raise NotImplementedError("Subclasses must implement calculate_control")

    def state_error(
        self,
        engine: PhysicsSimEngine,
        target_pose: BaseLinkPose,
        *,
        target_linear_velocity_world_m_s: Optional[np.ndarray] = None,
        target_angular_velocity_body_rad_s: Optional[np.ndarray] = None,
    ) -> StateError:
        """Compute the current-vs-target error.

        Translation lives in the world frame and rotation lives in the body
        (``base_link``) frame, matching :class:`StateError`'s conventions and
        the frame in which :class:`PhysicsSimEngine` stores angular velocity.
        The rotation error is computed as ``q_current^-1 * q_target`` so it
        represents the body-frame rotation that takes the current orientation
        to the target. The optional velocity targets default to zero, which is
        correct for a pose-hold task; pass them for trajectory tracking.
        """
        current = engine.base_pose

        position_err_world = (
            np.asarray(target_pose.position_m, dtype=float).reshape(3)
            - current.position_m
        )

        R_current = rotation_body_to_world_from_quat_xyzw(current.quaternion_xyzw)
        R_target = rotation_body_to_world_from_quat_xyzw(target_pose.quaternion_xyzw)
        R_err_body = R_current.T @ R_target
        rotation_err_body = _rotation_vector_from_matrix(R_err_body)

        tgt_lin = (
            np.zeros(3, dtype=float)
            if target_linear_velocity_world_m_s is None
            else np.asarray(target_linear_velocity_world_m_s, dtype=float).reshape(3)
        )
        tgt_ang = (
            np.zeros(3, dtype=float)
            if target_angular_velocity_body_rad_s is None
            else np.asarray(target_angular_velocity_body_rad_s, dtype=float).reshape(3)
        )

        return StateError(
            position_error_world_m=position_err_world,
            rotation_error_body_rad=rotation_err_body,
            linear_velocity_error_world_m_s=tgt_lin - engine.linear_velocity_world_m_s,
            angular_velocity_error_body_rad_s=tgt_ang - engine.angular_velocity_body_rad_s,
        )

    def thruster_IK_simple(
        self,
        engine: PhysicsSimEngine,
        desired_force_base: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Direction-only IK for a pure-force wrench (no commanded torque).

        Solves :func:`robot_description.ik.solve_arm_ik` independently for each
        arm with the desired body-frame thrust direction as the only task,
        seeded from the current joint configuration so the solver picks a
        nearby pose. Both thrusters end up pointed along the same body-frame
        direction; the lever-arm contributions are not explicitly cancelled
        here, so a small residual torque can persist (acceptable for an MVP
        position controller).

        Returns the combined 8-vector ``[left j1..j4, right j1..j4]`` matching
        :attr:`PhysicsSimEngine.joint_positions_rad`. The solver's
        ``direction_tolerance_rad`` is strict (``1e-4``); for joint
        configurations pinned at a URDF limit the residual direction error
        may exceed that tolerance, but the best-effort joints returned by
        :func:`solve_arm_ik` are still close enough for bang-bang use - the
        controller corrects on subsequent ticks. Only zero-magnitude input
        returns ``None``.
        """
        force = np.asarray(desired_force_base, dtype=float).reshape(3)
        n = float(np.linalg.norm(force))
        if n < 1e-12:
            return None
        direction = force / n

        q_now = engine.joint_positions_rad
        res_left = solve_arm_ik(
            engine.left_arm,
            target_thrust_direction_base=direction,
            initial_joint_positions_rad=q_now[0:4],
        )
        res_right = solve_arm_ik(
            engine.right_arm,
            target_thrust_direction_base=direction,
            initial_joint_positions_rad=q_now[4:8],
        )

        return np.concatenate(
            [res_left.joint_positions_rad, res_right.joint_positions_rad]
        )

    def thruster_IK(
        self,
        engine: PhysicsSimEngine,
        desired_force_base: np.ndarray,
        desired_torque_base: np.ndarray,
    ) -> np.ndarray:
        """Position + orientation IK for an arbitrary (force, torque) wrench.

        Pick a thruster position + orientation for each arm such that the
        combined thrust line and lever arm produce the requested force and
        torque (subject to actuator authority and joint limits). Delegates
        the per-arm joint search to :func:`robot_description.ik.solve_arm_ik`.

        Returns the combined 8-vector ``[left j1..j4, right j1..j4]`` matching
        :attr:`PhysicsSimEngine.joint_positions_rad`.
        """
        raise NotImplementedError("thruster_IK implementation pending")
