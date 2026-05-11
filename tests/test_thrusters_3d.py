"""Thruster FK in base frame for 3D viz."""

from __future__ import annotations

import numpy as np

from physics_sim import PhysicsSimEngine
from robot_description.fk import arm_link_frame_origins_base_m
from physics_sim.thrusters_3d import thruster_origin_and_thrust_axis_base


def test_thruster_origin_and_axis_are_unit_and_in_base_frame() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q_left = engine.joint_positions_rad[0:4]
    o, d = thruster_origin_and_thrust_axis_base(engine.left_arm, q_left)
    assert o.shape == (3,) and d.shape == (3,)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert np.linalg.norm(o) > 0.01


def test_arm_link_frame_origins_shape_and_base_at_zero() -> None:
    engine = PhysicsSimEngine.from_urdf()
    q_left = engine.joint_positions_rad[0:4]
    pts = arm_link_frame_origins_base_m(engine.left_arm, q_left)
    assert pts.shape == (5, 3)
    assert np.allclose(pts[0], 0.0)
    assert np.linalg.norm(pts[4] - pts[0]) > 0.01
