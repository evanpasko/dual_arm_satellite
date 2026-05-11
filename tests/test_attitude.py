"""Quaternion integration with zero body rate."""

from __future__ import annotations

import numpy as np

from physics_sim.attitude import integrate_quaternion_semi_implicit


def test_integrate_quaternion_zero_rate_preserves_identity() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    w = np.zeros(3)
    q1 = integrate_quaternion_semi_implicit(q0, w, 0.05)
    assert np.allclose(q1, q0, atol=1e-14)
    assert abs(np.linalg.norm(q1) - 1.0) < 1e-14
