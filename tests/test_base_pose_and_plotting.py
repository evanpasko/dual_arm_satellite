"""Start pose, logging, and plotting utilities."""

from __future__ import annotations

import numpy as np

from physics_sim import (
    BaseLinkPose,
    PhysicsSimEngine,
    normalize_quaternion_xyzw,
    plot_base_link_pose,
    plot_pose_log,
)


def test_optional_start_pose_and_pose_log() -> None:
    p0 = BaseLinkPose(
        np.array([1.0, -0.5, 0.25]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    q_init = np.linspace(0.0, 0.1, 8)
    engine = PhysicsSimEngine.from_urdf(
        initial_base_pose=p0,
        initial_joint_positions_rad=q_init,
    )
    assert np.allclose(engine.base_pose.position_m, p0.position_m)
    assert np.allclose(engine.base_pose.quaternion_xyzw, p0.quaternion_xyzw)
    assert np.allclose(engine.joint_positions_rad, q_init)

    engine.record_base_pose(0.0)
    engine.set_base_pose(
        BaseLinkPose(
            np.array([1.1, -0.4, 0.3]),
            normalize_quaternion_xyzw(np.array([0.1, 0.0, 0.0, 1.0])),
        )
    )
    engine.record_base_pose(0.1)
    assert len(engine.pose_log) == 2
    t, pos, quat = engine.pose_log.as_arrays()
    assert t.shape == (2,) and pos.shape == (2, 3) and quat.shape == (2, 4)


def test_plot_base_link_pose_to_file(tmp_path) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")

    t = np.array([0.0, 0.5, 1.0])
    p = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0.05, 0.0]])
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    out = tmp_path / "pose.png"
    fig = plot_base_link_pose(t, p, q, save_path=out, show=False)
    assert out.is_file()
    assert fig is not None


def test_plot_pose_log_wrapper(tmp_path) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")

    engine = PhysicsSimEngine.from_urdf()
    for k in range(5):
        engine.record_base_pose(0.02 * k)
        engine.set_base_pose(
            BaseLinkPose(
                np.array([0.01 * k, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )
    engine.record_base_pose(0.1)
    out = tmp_path / "log.png"
    plot_pose_log(engine.pose_log, save_path=out, show=False)
    assert out.is_file()
