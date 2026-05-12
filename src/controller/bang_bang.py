"""Bang-bang pose controller: each thruster fires at full throttle or holds off."""

from __future__ import annotations

from controller.base import ControllerBaseClass
from controller.types import ControlCommand
from physics_sim import BaseLinkPose, PhysicsSimEngine
from robot_description.fk import rotation_body_to_world_from_quat_xyzw


class BangBangController(ControllerBaseClass):
    """Sign-of-error actuation. Throttle is clamped to ``{0, 1}`` per side.

    MVP scope: position only. Each tick:

    1. Compute world-frame position error.
    2. If within the deadband, hold current joints and don't fire.
    3. Otherwise, point both thrusters along the world-to-target direction
       (rotated into body frame for the IK) and fire both at full throttle.

    Orientation, lever-arm-induced residual torque, and angular damping are
    not handled here yet.
    """

    def __init__(
        self,
        max_thrust_n: float,
        dist_err_threshold_m: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_thrust_n = max_thrust_n
        self.dist_err_threshold_m = dist_err_threshold_m

    def calculate_control(
        self,
        engine: PhysicsSimEngine,
        target_pose: BaseLinkPose,
    ) -> ControlCommand:
        err = self.state_error(engine, target_pose)
        dist_err = err.position_error_norm_m
        if dist_err < self.dist_err_threshold_m:
            return ControlCommand.hold(engine.joint_positions_rad)

        # Thrust along world->target direction. The IK takes a body-frame
        # vector, so rotate by R_world_to_body = R_body_to_world.T.
        force_dir_world = err.position_error_world_m / dist_err
        R_body_to_world = rotation_body_to_world_from_quat_xyzw(
            engine.base_pose.quaternion_xyzw
        )
        force_dir_body = R_body_to_world.T @ force_dir_world

        joints = self.thruster_IK_simple(engine, force_dir_body)
        if joints is None:
            return ControlCommand.hold(engine.joint_positions_rad)

        return ControlCommand(
            joint_positions_rad=joints,
            left_throttle=1.0,
            right_throttle=1.0,
        )
