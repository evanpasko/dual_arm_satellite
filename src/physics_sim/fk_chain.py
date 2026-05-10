"""Lightweight forward kinematics for arm chains (URDF-style transforms)."""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

from physics_sim.base_state import BaseLinkPose
from physics_sim.models import FixedJointParams, RevoluteJointParams, RobotArmDefinition


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _rodrigues(axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed rotation by ``angle_rad`` about ``axis_unit``."""
    a = np.asarray(axis_unit, dtype=float).reshape(3)
    K = _skew(a)
    s, c = math.sin(angle_rad), math.cos(angle_rad)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """URDF / ROS fixed-axis roll–pitch–yaw: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=float).reshape(3)]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def _homogeneous_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return T


def _revolute_transform(j: RevoluteJointParams, q_rad: float) -> np.ndarray:
    """Parent link → child link transform for angle ``q_rad``."""
    T0 = _homogeneous_from_xyz_rpy(
        np.array(j.origin_xyz_m), np.array(j.origin_rpy_rad)
    )
    axis = np.asarray(j.axis, dtype=float).reshape(3)
    R = _rodrigues(axis, q_rad)
    R4 = np.eye(4, dtype=float)
    R4[:3, :3] = R
    return T0 @ R4


def _fixed_transform(f: FixedJointParams) -> np.ndarray:
    return _homogeneous_from_xyz_rpy(
        np.array(f.origin_xyz_m), np.array(f.origin_rpy_rad)
    )


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (x,y,z,w): rotation body → world, ``v_w = R @ v_b``."""
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=float).reshape(4)]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def T_world_base(pose: BaseLinkPose) -> np.ndarray:
    """World ← base_link: ``p_world = T @ p_base`` (homogeneous 4×4)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quat_xyzw_to_R(pose.quaternion_xyzw)
    T[:3, 3] = pose.position_m.reshape(3)
    return T


def T_base_thruster(arm: RobotArmDefinition, joint_positions_rad: np.ndarray) -> np.ndarray:
    """base_link ← thruster frame: ``p_base = T @ p_thruster``."""
    q = np.asarray(joint_positions_rad, dtype=float).reshape(4)
    T = np.eye(4, dtype=float)
    for j, qi in zip(arm.joints, q):
        T = T @ _revolute_transform(j, float(qi))
    T = T @ _fixed_transform(arm.thruster_mount)
    return T


def T_world_thruster(
    pose: BaseLinkPose, arm: RobotArmDefinition, joint_positions_rad: np.ndarray
) -> np.ndarray:
    """World ← thruster: ``p_world = T @ p_thruster``."""
    return T_world_base(pose) @ T_base_thruster(arm, joint_positions_rad)


def compute_thrust_axis_world(
    pose: BaseLinkPose,
    arm: RobotArmDefinition,
    joint_positions_rad: np.ndarray,
    *,
    thrust_axis_thruster: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unit vector in world frame along the thruster bore (+Y of thruster link by default).
    """
    if thrust_axis_thruster is None:
        axis_t = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        axis_t = np.asarray(thrust_axis_thruster, dtype=float).reshape(3)
    n = np.linalg.norm(axis_t)
    if n < 1e-15:
        raise ValueError("Thrust axis has zero length")
    axis_t = axis_t / n
    T_wt = T_world_thruster(pose, arm, joint_positions_rad)
    R_wt = T_wt[:3, :3]
    d = R_wt @ axis_t
    d = d / np.linalg.norm(d)
    return d


def arm_joint_slice(
    joint_positions_rad: np.ndarray, side: Literal["left", "right"]
) -> np.ndarray:
    q = np.asarray(joint_positions_rad, dtype=float).reshape(8)
    return q[0:4].copy() if side == "left" else q[4:8].copy()
