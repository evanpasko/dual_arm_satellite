"""Behaviour of ``ControllerBaseClass.state_error`` and ``thruster_IK_simple``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from controller.base import ControllerBaseClass
from physics_sim import BaseLinkPose, PhysicsSimEngine
from robot_description.fk import (
    arm_joint_slice,
    compute_thrust_axis_world,
    rotation_body_to_world_from_quat_xyzw,
)


def _quat_about_z(angle_rad: float) -> np.ndarray:
    half = 0.5 * angle_rad
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def _bare_controller() -> ControllerBaseClass:
    return ControllerBaseClass()


def test_state_error_identity_pose_is_zero() -> None:
    engine = PhysicsSimEngine.from_urdf()
    err = _bare_controller().state_error(engine, BaseLinkPose.identity())
    assert err.position_error_norm_m < 1e-12
    assert err.rotation_error_norm_rad < 1e-12
    assert err.linear_velocity_error_norm_m_s < 1e-12
    assert err.angular_velocity_error_norm_rad_s < 1e-12


def test_state_error_position_only_offset_in_world() -> None:
    engine = PhysicsSimEngine.from_urdf()
    target = BaseLinkPose(np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]))
    err = _bare_controller().state_error(engine, target)
    assert np.allclose(err.position_error_world_m, [1.0, -2.0, 0.5])
    assert err.rotation_error_norm_rad < 1e-12


def test_state_error_rotation_only_z_axis() -> None:
    engine = PhysicsSimEngine.from_urdf()
    angle = 0.3
    target = BaseLinkPose(np.zeros(3), _quat_about_z(angle))
    err = _bare_controller().state_error(engine, target)
    assert err.position_error_norm_m < 1e-12
    assert np.allclose(err.rotation_error_body_rad, [0.0, 0.0, angle], atol=1e-9)


def test_state_error_velocity_terms_use_engine_state_when_targets_zero() -> None:
    engine = PhysicsSimEngine.from_urdf()
    engine.set_linear_velocity_world_m_s(np.array([0.2, 0.0, -0.1]))
    engine.set_angular_velocity_body_rad_s(np.array([0.0, 0.05, 0.0]))
    err = _bare_controller().state_error(engine, BaseLinkPose.identity())
    assert np.allclose(err.linear_velocity_error_world_m_s, [-0.2, 0.0, 0.1])
    assert np.allclose(err.angular_velocity_error_body_rad_s, [0.0, -0.05, 0.0])


def test_state_error_velocity_targets_subtract_engine_state() -> None:
    engine = PhysicsSimEngine.from_urdf()
    engine.set_linear_velocity_world_m_s(np.array([0.2, 0.0, 0.0]))
    err = _bare_controller().state_error(
        engine,
        BaseLinkPose.identity(),
        target_linear_velocity_world_m_s=np.array([0.5, 0.0, 0.0]),
    )
    assert np.allclose(err.linear_velocity_error_world_m_s, [0.3, 0.0, 0.0])


def _world_thrust_direction(engine: PhysicsSimEngine, joints_8: np.ndarray, side: str) -> np.ndarray:
    arm = engine.left_arm if side == "left" else engine.right_arm
    q_side = arm_joint_slice(joints_8, side)
    return compute_thrust_axis_world(engine.base_pose, arm, q_side)


def test_thruster_IK_simple_points_both_arms_along_request() -> None:
    engine = PhysicsSimEngine.from_urdf()
    # Pick a body-frame direction reachable from default pose. The +Y default
    # is on the boundary, so nudge into a generic direction.
    desired_base = np.array([0.3, 1.0, 0.2])
    joints = _bare_controller().thruster_IK_simple(engine, desired_base)
    assert joints is not None
    assert joints.shape == (8,)

    R_b_to_w = rotation_body_to_world_from_quat_xyzw(engine.base_pose.quaternion_xyzw)
    desired_world = R_b_to_w @ (desired_base / np.linalg.norm(desired_base))

    d_left = _world_thrust_direction(engine, joints, "left")
    d_right = _world_thrust_direction(engine, joints, "right")
    assert math.acos(float(np.clip(d_left @ desired_world, -1.0, 1.0))) < 5e-3
    assert math.acos(float(np.clip(d_right @ desired_world, -1.0, 1.0))) < 5e-3


def test_thruster_IK_simple_zero_force_returns_none() -> None:
    engine = PhysicsSimEngine.from_urdf()
    assert _bare_controller().thruster_IK_simple(engine, np.zeros(3)) is None


def test_thruster_IK_simple_returns_eight_vector_concatenation() -> None:
    engine = PhysicsSimEngine.from_urdf()
    joints = _bare_controller().thruster_IK_simple(engine, np.array([1.0, 0.0, 0.5]))
    assert joints is not None and joints.shape == (8,)
    # Sliced halves are valid 4-vectors usable as arm joint states.
    assert arm_joint_slice(joints, "left").shape == (4,)
    assert arm_joint_slice(joints, "right").shape == (4,)
