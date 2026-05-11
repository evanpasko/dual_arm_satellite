"""Thruster bore direction (+Y) and linear impulse at CoM."""

from __future__ import annotations

import numpy as np

from physics_sim import PhysicsSimEngine, Thruster
from physics_sim.fk_chain import compute_thrust_axis_world


def test_thrust_axis_unit_and_matches_fk_helper() -> None:
    engine = PhysicsSimEngine.from_urdf()
    lt = Thruster(side="left", max_thrust_n=10.0)
    d = lt.thrust_direction_world(engine)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    q_left = engine.joint_positions_rad[0:4]
    d2 = compute_thrust_axis_world(
        engine.base_pose,
        engine.left_arm,
        q_left,
        thrust_axis_thruster=Thruster.thrust_axis_thruster_frame(),
    )
    assert np.allclose(d, d2)


def test_fire_impulse_updates_linear_velocity() -> None:
    engine = PhysicsSimEngine.from_urdf(
        left_thruster=Thruster(side="left", max_thrust_n=800.0, impulse_window_s=0.1),
    )
    assert np.allclose(engine.linear_velocity_world_m_s, 0.0)
    assert np.allclose(engine.angular_velocity_body_rad_s, 0.0)
    J = engine.left_thruster.fire(engine, throttle=1.0)
    m = engine.satellite.mass_kg
    expected_delta_v = J / m
    assert np.allclose(engine.linear_velocity_world_m_s, expected_delta_v)
    assert np.isclose(np.linalg.norm(J), 800.0 * 0.1)
    # Lever arm from CoM to EE produces body torque (default URDF pose).
    assert np.linalg.norm(engine.angular_velocity_body_rad_s) > 1e-12


def test_throttle_scales_impulse_magnitude() -> None:
    engine = PhysicsSimEngine.from_urdf(
        right_thruster=Thruster(side="right", max_thrust_n=100.0, impulse_window_s=0.02),
    )
    d = engine.right_thruster.thrust_direction_world(engine)
    J_half = engine.right_thruster.fire(engine, throttle=0.5)
    assert np.isclose(np.linalg.norm(J_half), 50.0 * 0.02)
    assert np.allclose(J_half / np.linalg.norm(J_half), d)
