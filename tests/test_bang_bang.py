"""Bang-bang MVP: deadband hold and thrust-toward-target behaviour."""

from __future__ import annotations

import math

import numpy as np

from controller import BangBangController, ControlCommand
from physics_sim import BaseLinkPose, PhysicsSimEngine
from robot_description.fk import (
    arm_joint_slice,
    compute_thrust_axis_world,
)


def test_calculate_control_within_deadband_holds() -> None:
    engine = PhysicsSimEngine.from_urdf()
    controller = BangBangController(max_thrust_n=1.0, dist_err_threshold_m=0.05)
    target = BaseLinkPose(np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    cmd = controller.calculate_control(engine, target)
    assert isinstance(cmd, ControlCommand)
    assert cmd.left_throttle == 0.0 and cmd.right_throttle == 0.0
    assert np.allclose(cmd.joint_positions_rad, engine.joint_positions_rad)


def test_calculate_control_outside_deadband_fires_both_full() -> None:
    engine = PhysicsSimEngine.from_urdf()
    controller = BangBangController(max_thrust_n=1.0)
    target = BaseLinkPose(np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    cmd = controller.calculate_control(engine, target)
    assert cmd.left_throttle == 1.0 and cmd.right_throttle == 1.0


def test_calculate_control_thrust_points_world_to_target() -> None:
    engine = PhysicsSimEngine.from_urdf()
    controller = BangBangController(max_thrust_n=1.0)
    target = BaseLinkPose(np.array([1.5, 0.3, -0.2]), np.array([0.0, 0.0, 0.0, 1.0]))
    cmd = controller.calculate_control(engine, target)
    assert cmd.is_left_firing and cmd.is_right_firing

    expected_world = target.position_m / np.linalg.norm(target.position_m)
    for side in ("left", "right"):
        d_world = compute_thrust_axis_world(
            engine.base_pose,
            engine.left_arm if side == "left" else engine.right_arm,
            arm_joint_slice(cmd.joint_positions_rad, side),
        )
        cos_th = float(np.clip(d_world @ expected_world, -1.0, 1.0))
        assert math.acos(cos_th) < 5e-3


def test_calculate_control_returns_eight_vector_joints() -> None:
    engine = PhysicsSimEngine.from_urdf()
    controller = BangBangController(max_thrust_n=1.0)
    target = BaseLinkPose(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    cmd = controller.calculate_control(engine, target)
    assert cmd.joint_positions_rad.shape == (8,)


def test_calculate_control_holds_when_state_already_on_target() -> None:
    engine = PhysicsSimEngine.from_urdf()
    controller = BangBangController(max_thrust_n=1.0)
    cmd = controller.calculate_control(engine, BaseLinkPose.identity())
    assert not cmd.is_left_firing and not cmd.is_right_firing
