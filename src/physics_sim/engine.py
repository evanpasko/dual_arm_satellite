"""Primary physics simulation engine facade."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from physics_sim.base_state import (
    BaseLinkPose,
    PoseTimeSeries,
    make_initial_joint_positions,
)
from physics_sim.models import ParsedRobotDescription, RobotArmDefinition, SatelliteBodyParams
from physics_sim.thruster import Thruster
from physics_sim.urdf_loading import default_urdf_path, parse_robot_urdf

UrdfPath = Union[str, Path]


class PhysicsSimEngine:
    """
    Owns satellite bus parameters and both arm definitions, typically built from the URDF.

    Tracks an optional initial ``base_link`` pose and 8 joint angles for teleop / control.
    Linear velocity of the bus CoM in world frame can be updated via impulses (see thrusters).
    Use :meth:`record_base_pose` during integration to build data for :mod:`physics_sim.plotting`.
    """

    def __init__(
        self,
        satellite: SatelliteBodyParams,
        left_arm: RobotArmDefinition,
        right_arm: RobotArmDefinition,
        *,
        urdf_path: Optional[UrdfPath] = None,
        initial_base_pose: Optional[BaseLinkPose] = None,
        initial_joint_positions_rad: Optional[np.ndarray] = None,
        left_thruster: Optional[Thruster] = None,
        right_thruster: Optional[Thruster] = None,
    ) -> None:
        self.satellite = satellite
        self.left_arm = left_arm
        self.right_arm = right_arm
        self._urdf_path: Optional[Path] = Path(urdf_path) if urdf_path is not None else None

        self._base_pose: BaseLinkPose = (
            initial_base_pose.copy()
            if initial_base_pose is not None
            else BaseLinkPose.identity()
        )
        self._joint_positions_rad: np.ndarray = make_initial_joint_positions()
        if initial_joint_positions_rad is not None:
            self._joint_positions_rad = np.asarray(
                initial_joint_positions_rad, dtype=float
            ).reshape(8)

        self._linear_velocity_world_m_s = np.zeros(3, dtype=float)
        self.left_thruster = (
            left_thruster
            if left_thruster is not None
            else Thruster(side="left", max_thrust_n=1.0)
        )
        self.right_thruster = (
            right_thruster
            if right_thruster is not None
            else Thruster(side="right", max_thrust_n=1.0)
        )

        self.pose_log = PoseTimeSeries()

    @property
    def urdf_path(self) -> Optional[Path]:
        return self._urdf_path

    @property
    def base_pose(self) -> BaseLinkPose:
        return self._base_pose.copy()

    def set_base_pose(self, pose: BaseLinkPose) -> None:
        self._base_pose = pose.copy()

    @property
    def joint_positions_rad(self) -> np.ndarray:
        return self._joint_positions_rad.copy()

    def set_joint_positions_rad(self, q: np.ndarray) -> None:
        self._joint_positions_rad = np.asarray(q, dtype=float).reshape(8)

    @property
    def linear_velocity_world_m_s(self) -> np.ndarray:
        """Translational velocity of the bus CoM in world frame (m/s)."""
        return self._linear_velocity_world_m_s.copy()

    def set_linear_velocity_world_m_s(self, v: np.ndarray) -> None:
        self._linear_velocity_world_m_s = np.asarray(v, dtype=float).reshape(3).copy()

    def step(self, dt_s: float) -> None:
        """
        Advance ``base_link`` translation by one step: ``p += v * dt`` (world frame).

        Orientation is unchanged (no angular state in this MVP). ``dt_s`` must be non-negative.
        """
        dt = float(dt_s)
        if dt < 0.0:
            raise ValueError("dt_s must be non-negative")
        p = self._base_pose.position_m + self._linear_velocity_world_m_s * dt
        q = self._base_pose.quaternion_xyzw.copy()
        self._base_pose = BaseLinkPose(p, q)

    def apply_linear_impulse_world(self, impulse_n_s: np.ndarray) -> None:
        """
        Apply an instantaneous linear impulse at the satellite CoM (world frame, N·s).

        ``Δv = J / m`` with :attr:`satellite.mass_kg`. No angular impulse (massless arms).
        """
        J = np.asarray(impulse_n_s, dtype=float).reshape(3)
        m = self.satellite.mass_kg
        if m <= 0.0:
            raise ValueError("Satellite mass must be positive to apply impulse")
        self._linear_velocity_world_m_s = self._linear_velocity_world_m_s + J / m

    def reset(
        self,
        initial_base_pose: Optional[BaseLinkPose] = None,
        initial_joint_positions_rad: Optional[np.ndarray] = None,
        *,
        clear_pose_log: bool = True,
    ) -> None:
        """Restore optional start state; by default clears recorded pose history."""
        if initial_base_pose is None:
            self._base_pose = BaseLinkPose.identity()
        else:
            self._base_pose = initial_base_pose.copy()
        if initial_joint_positions_rad is None:
            self._joint_positions_rad = make_initial_joint_positions()
        else:
            self._joint_positions_rad = np.asarray(
                initial_joint_positions_rad, dtype=float
            ).reshape(8)
        if clear_pose_log:
            self.pose_log.clear()
        self._linear_velocity_world_m_s = np.zeros(3, dtype=float)

    def record_base_pose(self, time_s: float) -> None:
        """Append the current ``base_link`` pose to :attr:`pose_log` at ``time_s`` seconds."""
        self.pose_log.append(time_s, self._base_pose)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Optional[UrdfPath] = None,
        *,
        initial_base_pose: Optional[BaseLinkPose] = None,
        initial_joint_positions_rad: Optional[np.ndarray] = None,
        left_thruster: Optional[Thruster] = None,
        right_thruster: Optional[Thruster] = None,
    ) -> PhysicsSimEngine:
        """Load ``ParsedRobotDescription`` from disk and construct the engine."""
        path = default_urdf_path() if urdf_path is None else Path(urdf_path)
        parsed = parse_robot_urdf(path)
        return cls(
            parsed.satellite,
            parsed.left_arm,
            parsed.right_arm,
            urdf_path=path,
            initial_base_pose=initial_base_pose,
            initial_joint_positions_rad=initial_joint_positions_rad,
            left_thruster=left_thruster,
            right_thruster=right_thruster,
        )

    @staticmethod
    def load_description(urdf_path: Optional[UrdfPath] = None) -> ParsedRobotDescription:
        """Parse URDF only (no engine), useful for tests or tooling."""
        path = default_urdf_path() if urdf_path is None else Path(urdf_path)
        return parse_robot_urdf(path)
