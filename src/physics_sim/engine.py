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
from physics_sim.urdf_loading import default_urdf_path, parse_robot_urdf

UrdfPath = Union[str, Path]


class PhysicsSimEngine:
    """
    Owns satellite bus parameters and both arm definitions, typically built from the URDF.

    Tracks an optional initial ``base_link`` pose and 8 joint angles for teleop / control.
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
        )

    @staticmethod
    def load_description(urdf_path: Optional[UrdfPath] = None) -> ParsedRobotDescription:
        """Parse URDF only (no engine), useful for tests or tooling."""
        path = default_urdf_path() if urdf_path is None else Path(urdf_path)
        return parse_robot_urdf(path)
