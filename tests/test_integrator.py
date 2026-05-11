"""Constant-acceleration pose integrator."""

from __future__ import annotations

import numpy as np

from physics_sim.integrator import integrate_pose_constant_accel


def test_integrator_quaternion_stays_unit() -> None:
    t, p, q = integrate_pose_constant_accel(
        duration_s=5.0,
        dt_s=0.01,
        position0_m=np.zeros(3),
        quaternion_xyzw0=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity_world_m_s=np.array([0.01, 0.0, 0.0]),
        angular_velocity_body_rad_s=np.array([0.0, 0.02, 0.0]),
        acceleration_world_m_s2=np.array([0.001, 0.0, 0.0]),
        angular_accel_body_rad_s2=np.array([0.0001, 0.0, 0.0]),
    )
    norms = np.linalg.norm(q, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert t[0] == 0.0 and abs(t[-1] - 5.0) < 1e-9
    assert p.shape[0] == q.shape[0] == t.shape[0]


def test_integrator_linear_motion_no_spin() -> None:
    t, p, q = integrate_pose_constant_accel(
        duration_s=2.0,
        dt_s=0.1,
        position0_m=np.zeros(3),
        quaternion_xyzw0=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity_world_m_s=np.zeros(3),
        angular_velocity_body_rad_s=np.zeros(3),
        acceleration_world_m_s2=np.array([1.0, 0.0, 0.0]),
        angular_accel_body_rad_s2=np.zeros(3),
    )
    # p_x = 0.5 * a * t^2 at end (v0=0, constant a; discrete semi-implicit approximates this)
    assert p[-1, 0] > 1.9  # close to 2.0
    assert np.allclose(q[-1], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
