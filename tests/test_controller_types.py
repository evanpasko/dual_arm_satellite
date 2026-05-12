"""Value-type guarantees for ``StateError`` and ``ControlCommand``."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from controller import ControlCommand, StateError


def _err(
    p=(0.1, -0.2, 0.3),
    r=(0.0, 0.0, 0.05),
    v=(0.0, 0.0, 0.0),
    w=(0.0, 0.0, 0.0),
) -> StateError:
    return StateError(
        position_error_world_m=np.asarray(p, dtype=float),
        rotation_error_body_rad=np.asarray(r, dtype=float),
        linear_velocity_error_world_m_s=np.asarray(v, dtype=float),
        angular_velocity_error_body_rad_s=np.asarray(w, dtype=float),
    )


def test_state_error_stores_arrays_and_reports_norms() -> None:
    e = _err(p=(3.0, 4.0, 0.0), r=(0.0, 0.0, 0.1), v=(0.0, 1.0, 0.0), w=(0.2, 0.0, 0.0))
    assert e.position_error_world_m.shape == (3,)
    assert math.isclose(e.position_error_norm_m, 5.0)
    assert math.isclose(e.rotation_error_norm_rad, 0.1)
    assert math.isclose(e.linear_velocity_error_norm_m_s, 1.0)
    assert math.isclose(e.angular_velocity_error_norm_rad_s, 0.2)


def test_state_error_zero_factory_is_all_zero() -> None:
    e = StateError.zero()
    assert e.position_error_norm_m == 0.0
    assert e.rotation_error_norm_rad == 0.0
    assert e.linear_velocity_error_norm_m_s == 0.0
    assert e.angular_velocity_error_norm_rad_s == 0.0


def test_state_error_is_frozen() -> None:
    e = _err()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.position_error_world_m = np.zeros(3)  # type: ignore[misc]


def test_state_error_arrays_are_read_only_and_defensively_copied() -> None:
    src = np.array([0.1, 0.2, 0.3], dtype=float)
    e = StateError(
        position_error_world_m=src,
        rotation_error_body_rad=np.zeros(3),
        linear_velocity_error_world_m_s=np.zeros(3),
        angular_velocity_error_body_rad_s=np.zeros(3),
    )
    src[0] = 99.0
    assert e.position_error_world_m[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        e.position_error_world_m[0] = 1.0


def test_state_error_accepts_list_inputs_via_reshape() -> None:
    e = StateError(
        position_error_world_m=[1.0, 2.0, 3.0],
        rotation_error_body_rad=[0.0, 0.0, 0.0],
        linear_velocity_error_world_m_s=[0.0, 0.0, 0.0],
        angular_velocity_error_body_rad_s=[0.0, 0.0, 0.0],
    )
    assert e.position_error_world_m.tolist() == [1.0, 2.0, 3.0]


def test_state_error_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        StateError(
            position_error_world_m=[1.0, 2.0],
            rotation_error_body_rad=np.zeros(3),
            linear_velocity_error_world_m_s=np.zeros(3),
            angular_velocity_error_body_rad_s=np.zeros(3),
        )


def test_state_error_tolerance_pose_only() -> None:
    e = _err(p=(0.01, 0.0, 0.0), r=(0.0, 0.001, 0.0), v=(10.0, 0.0, 0.0), w=(10.0, 0.0, 0.0))
    assert e.is_within_tolerance(position_tolerance_m=0.1, rotation_tolerance_rad=0.01)


def test_state_error_tolerance_with_velocity_caps() -> None:
    e = _err(p=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0), v=(0.5, 0.0, 0.0), w=(0.0, 0.0, 0.0))
    assert not e.is_within_tolerance(
        position_tolerance_m=0.1,
        rotation_tolerance_rad=0.01,
        linear_velocity_tolerance_m_s=0.1,
    )
    assert e.is_within_tolerance(
        position_tolerance_m=0.1,
        rotation_tolerance_rad=0.01,
        linear_velocity_tolerance_m_s=1.0,
    )


def _cmd(
    joints=(0.0,) * 8,
    left=0.0,
    right=0.0,
) -> ControlCommand:
    return ControlCommand(
        joint_positions_rad=np.asarray(joints, dtype=float),
        left_throttle=left,
        right_throttle=right,
    )


def test_control_command_stores_joints_and_throttles() -> None:
    c = _cmd(joints=(0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4), left=0.5, right=1.0)
    assert c.joint_positions_rad.shape == (8,)
    assert c.left_throttle == 0.5
    assert c.right_throttle == 1.0
    assert np.allclose(c.left_joint_positions_rad, [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(c.right_joint_positions_rad, [-0.1, -0.2, -0.3, -0.4])


def test_control_command_throttle_validation() -> None:
    with pytest.raises(ValueError):
        _cmd(left=-0.01)
    with pytest.raises(ValueError):
        _cmd(right=1.0001)


def test_control_command_joint_shape_validation() -> None:
    with pytest.raises(ValueError):
        ControlCommand(
            joint_positions_rad=np.zeros(4),
            left_throttle=0.0,
            right_throttle=0.0,
        )


def test_control_command_is_frozen_and_joints_are_read_only() -> None:
    c = _cmd()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.left_throttle = 0.5  # type: ignore[misc]
    with pytest.raises(ValueError):
        c.joint_positions_rad[0] = 1.0


def test_control_command_hold_factory_is_zero_throttle() -> None:
    q = np.linspace(-0.4, 0.4, 8)
    c = ControlCommand.hold(joint_positions_rad=q)
    assert np.allclose(c.joint_positions_rad, q)
    assert c.left_throttle == 0.0 and c.right_throttle == 0.0
    assert not c.is_left_firing and not c.is_right_firing


def test_control_command_firing_flags() -> None:
    assert _cmd(left=0.0, right=0.5).is_right_firing
    assert not _cmd(left=0.0, right=0.5).is_left_firing
