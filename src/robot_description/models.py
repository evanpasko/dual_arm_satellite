"""Parameter objects describing the robot (populated from the URDF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Inertia:
    """Spatial inertia of a link about its inertial frame (URDF convention)."""

    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass(frozen=True)
class SatelliteBodyParams:
    """Rigid satellite bus: maps to URDF ``base_link`` (disk / main body)."""

    link_name: str
    mass_kg: float
    inertia: Inertia
    com_xyz_m: Vec3
    cylinder_radius_m: float
    cylinder_length_m: float


@dataclass(frozen=True)
class RevoluteJointParams:
    """One-DOF hinge between parent and child link."""

    name: str
    parent_link: str
    child_link: str
    origin_xyz_m: Vec3
    origin_rpy_rad: Vec3
    axis: Vec3
    limit_lower_rad: float
    limit_upper_rad: float


@dataclass(frozen=True)
class FixedJointParams:
    """Constant transform from parent link to child link."""

    name: str
    parent_link: str
    child_link: str
    origin_xyz_m: Vec3
    origin_rpy_rad: Vec3


@dataclass(frozen=True)
class ArmLinkCylinderParams:
    """Primitive cylinder geometry for one arm segment (from URDF visual)."""

    link_name: str
    length_m: float
    radius_m: float


@dataclass(frozen=True)
class RobotArmDefinition:
    """One 4-DoF arm: kinematic chain + thruster mount, derived from URDF."""

    side: Literal["left", "right"]
    joints: Tuple[
        RevoluteJointParams,
        RevoluteJointParams,
        RevoluteJointParams,
        RevoluteJointParams,
    ]
    thruster_mount: FixedJointParams
    link_cylinders: Tuple[
        ArmLinkCylinderParams,
        ArmLinkCylinderParams,
        ArmLinkCylinderParams,
        ArmLinkCylinderParams,
    ]


@dataclass(frozen=True)
class ParsedRobotDescription:
    """Everything the sim needs from ``dual_arm_satellite.urdf``."""

    satellite: SatelliteBodyParams
    left_arm: RobotArmDefinition
    right_arm: RobotArmDefinition
