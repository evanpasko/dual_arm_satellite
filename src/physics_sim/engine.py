"""Primary physics simulation engine facade."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from physics_sim.attitude import integrate_quaternion_semi_implicit
from physics_sim.base_state import (
    BaseLinkPose,
    PoseTimeSeries,
    make_initial_joint_positions,
)
from robot_description.fk import rotation_body_to_world_from_quat_xyzw
from physics_sim.inertia_utils import (
    inertia_tensor_body_from_satellite,
    invert_inertia_3x3,
)
from physics_sim.thruster import Thruster
from robot_description.models import (
    ParsedRobotDescription,
    RobotArmDefinition,
    SatelliteBodyParams,
)
from robot_description.urdf_loading import default_urdf_path, parse_robot_urdf

UrdfPath = Union[str, Path]


class PhysicsSimEngine:
    """
    Owns satellite bus parameters and both arm definitions, typically built from the URDF.

    Tracks ``base_link`` pose, linear velocity (world), **angular velocity (body)**, and 8
    joint angles. Thruster impulses apply **linear** momentum at the CoM and **angular**
    impulse ``r × J`` about the CoM when applied at the end-effector (URDF thruster origin).
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
        self._angular_velocity_body_rad_s = np.zeros(3, dtype=float)
        I_body = inertia_tensor_body_from_satellite(satellite)
        self._inertia_body_inv = invert_inertia_3x3(I_body)

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

    @property
    def angular_velocity_body_rad_s(self) -> np.ndarray:
        """Angular velocity of ``base_link`` expressed in the body frame (rad/s)."""
        return self._angular_velocity_body_rad_s.copy()

    def set_angular_velocity_body_rad_s(self, w: np.ndarray) -> None:
        self._angular_velocity_body_rad_s = np.asarray(w, dtype=float).reshape(3).copy()

    def step(self, dt_s: float) -> None:
        """
        Semi-implicit step: ``p += v dt``, quaternion from body-frame ``ω``.

        ``dt_s`` must be non-negative.
        """
        dt = float(dt_s)
        if dt < 0.0:
            raise ValueError("dt_s must be non-negative")
        p = self._base_pose.position_m + self._linear_velocity_world_m_s * dt
        q = integrate_quaternion_semi_implicit(
            self._base_pose.quaternion_xyzw,
            self._angular_velocity_body_rad_s,
            dt,
        )
        self._base_pose = BaseLinkPose(p, q)

    def apply_impulse_world(
        self,
        impulse_world_n_s: np.ndarray,
        *,
        application_point_world_m: Optional[np.ndarray] = None,
    ) -> None:
        """
        Apply a linear impulse ``J`` (N·s) in world frame at the CoM for linear momentum.

        If ``application_point_world_m`` is set (world-frame point, m), also applies the
        angular impulse ``ΔL = r × J`` with ``r`` from CoM to that point, updating body-frame
        ``ω`` via ``Δω = I^{-1} (r_b × J_b)``.
        """
        J = np.asarray(impulse_world_n_s, dtype=float).reshape(3)
        m = self.satellite.mass_kg
        if m <= 0.0:
            raise ValueError("Satellite mass must be positive to apply impulse")
        self._linear_velocity_world_m_s = self._linear_velocity_world_m_s + J / m

        if application_point_world_m is None:
            return

        p_com = self._base_pose.position_m.reshape(3)
        R = rotation_body_to_world_from_quat_xyzw(self._base_pose.quaternion_xyzw)
        p_app = np.asarray(application_point_world_m, dtype=float).reshape(3)
        r_w = p_app - p_com
        r_b = R.T @ r_w
        J_b = R.T @ J
        tau_b = np.cross(r_b, J_b)
        domega = self._inertia_body_inv @ tau_b
        self._angular_velocity_body_rad_s = self._angular_velocity_body_rad_s + domega

    def apply_linear_impulse_world(self, impulse_n_s: np.ndarray) -> None:
        """Apply impulse at the CoM only (no torque). Same as ``apply_impulse_world(J)``."""
        self.apply_impulse_world(impulse_n_s, application_point_world_m=None)

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
        self._angular_velocity_body_rad_s = np.zeros(3, dtype=float)

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
