"""Translation-only physics step."""

from __future__ import annotations

import numpy as np

from physics_sim import BaseLinkPose, PhysicsSimEngine


def test_step_updates_position_from_linear_velocity() -> None:
    engine = PhysicsSimEngine.from_urdf(
        initial_base_pose=BaseLinkPose(
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ),
    )
    engine.set_linear_velocity_world_m_s(np.array([1.0, -0.5, 0.25]))
    engine.step(0.1)
    p = engine.base_pose.position_m
    assert np.allclose(p, [0.1, -0.05, 0.025])
    q = engine.base_pose.quaternion_xyzw
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])
