"""Damped least-squares IK round-trips for one 4-DoF arm in base_link frame."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics_sim import PhysicsSimEngine
from robot_description.fk import T_base_thruster
from robot_description.ik import (
    IKResult,
    solve_arm_ik,
    thruster_pose_in_base,
)


def _fk_pose(arm, q):
    return thruster_pose_in_base(arm, q)


def test_thruster_pose_in_base_matches_fk_transform() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q = np.array([0.3, -0.4, 0.5, 0.2])
    p, d = _fk_pose(engine.left_arm, q)
    T = T_base_thruster(engine.left_arm, q)
    assert np.allclose(p, T[:3, 3])
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert np.allclose(d, T[:3, :3] @ np.array([0.0, 1.0, 0.0]))


def test_ik_requires_at_least_one_target() -> None:
    engine = PhysicsSimEngine.from_urdf()
    with pytest.raises(ValueError):
        solve_arm_ik(engine.left_arm)


def test_ik_position_only_round_trip_left_arm() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q_truth = np.array([0.25, 0.3, -0.4, 0.1])
    p_target, _ = _fk_pose(engine.left_arm, q_truth)
    res: IKResult = solve_arm_ik(
        engine.left_arm,
        target_position_base_m=p_target,
        initial_joint_positions_rad=np.zeros(4),
        position_tolerance_m=1e-5,
    )
    assert res.converged, res
    p_solved, _ = _fk_pose(engine.left_arm, res.joint_positions_rad)
    assert np.linalg.norm(p_solved - p_target) < 1e-4


def test_ik_direction_only_round_trip_left_arm() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q_truth = np.array([0.1, 0.4, -0.2, 0.3])
    _, d_target = _fk_pose(engine.left_arm, q_truth)
    res = solve_arm_ik(
        engine.left_arm,
        target_thrust_direction_base=d_target,
        initial_joint_positions_rad=np.zeros(4),
        direction_tolerance_rad=1e-5,
    )
    assert res.converged, res
    _, d_solved = _fk_pose(engine.left_arm, res.joint_positions_rad)
    cos_th = float(np.clip(d_solved @ d_target, -1.0, 1.0))
    assert math.acos(cos_th) < 1e-4


def test_ik_position_and_direction_round_trip_right_arm() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q_truth = np.array([-0.2, 0.3, 0.45, -0.15])
    p_target, d_target = _fk_pose(engine.right_arm, q_truth)
    res = solve_arm_ik(
        engine.right_arm,
        target_position_base_m=p_target,
        target_thrust_direction_base=d_target,
        initial_joint_positions_rad=np.zeros(4),
        position_tolerance_m=1e-4,
        direction_tolerance_rad=1e-4,
        max_iterations=400,
    )
    assert res.converged, res
    p_solved, d_solved = _fk_pose(engine.right_arm, res.joint_positions_rad)
    assert np.linalg.norm(p_solved - p_target) < 5e-4
    cos_th = float(np.clip(d_solved @ d_target, -1.0, 1.0))
    assert math.acos(cos_th) < 1e-3


def test_ik_respects_joint_limits() -> None:
    engine = PhysicsSimEngine.from_urdf()
    p_far = np.array([5.0, 5.0, 5.0])
    res = solve_arm_ik(
        engine.left_arm,
        target_position_base_m=p_far,
        initial_joint_positions_rad=np.zeros(4),
        max_iterations=50,
    )
    for i, j in enumerate(engine.left_arm.joints):
        q_i = float(res.joint_positions_rad[i])
        assert j.limit_lower_rad - 1e-9 <= q_i <= j.limit_upper_rad + 1e-9


def test_ik_zero_target_direction_raises() -> None:
    engine = PhysicsSimEngine.from_urdf()
    with pytest.raises(ValueError):
        solve_arm_ik(
            engine.left_arm,
            target_thrust_direction_base=np.zeros(3),
        )
