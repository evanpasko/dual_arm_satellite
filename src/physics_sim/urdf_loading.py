"""Parse ``dual_arm_satellite.urdf`` into :class:`ParsedRobotDescription`."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple, cast

from physics_sim.models import (
    ArmLinkCylinderParams,
    FixedJointParams,
    Inertia,
    ParsedRobotDescription,
    RevoluteJointParams,
    RobotArmDefinition,
    SatelliteBodyParams,
    Vec3,
)

BASE_LINK_NAME = "base_link"


def _f(attr: Optional[str], label: str) -> float:
    if attr is None or attr == "":
        raise ValueError(f"Missing required numeric attribute: {label}")
    return float(attr)


def _parse_vec3(text: Optional[str], label: str) -> Vec3:
    if text is None:
        raise ValueError(f"Missing vector: {label}")
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"{label}: expected 3 floats, got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _norm_axis(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-12:
        raise ValueError("Joint axis has zero length")
    return (v[0] / n, v[1] / n, v[2] / n)


def _first_cylinder_from_link(link_el: ET.Element) -> Tuple[float, float]:
    visual = link_el.find("visual")
    if visual is None:
        raise ValueError(f"Link {link_el.get('name')!r} has no <visual> with cylinder")
    geom = visual.find("geometry")
    if geom is None:
        raise ValueError("Missing <geometry> under <visual>")
    cyl = geom.find("cylinder")
    if cyl is None:
        raise ValueError("Expected <cylinder> geometry for arm/bus link")
    return _f(cyl.get("length"), "cylinder@length"), _f(cyl.get("radius"), "cylinder@radius")


def _parse_inertia(el: ET.Element) -> Inertia:
    return Inertia(
        ixx=_f(el.get("ixx"), "ixx"),
        ixy=_f(el.get("ixy"), "ixy"),
        ixz=_f(el.get("ixz"), "ixz"),
        iyy=_f(el.get("iyy"), "iyy"),
        iyz=_f(el.get("iyz"), "iyz"),
        izz=_f(el.get("izz"), "izz"),
    )


def _parse_revolute(j_el: ET.Element) -> RevoluteJointParams:
    name = j_el.get("name")
    if not name:
        raise ValueError("Joint without name")
    parent_el = j_el.find("parent")
    child_el = j_el.find("child")
    origin_el = j_el.find("origin")
    axis_el = j_el.find("axis")
    limit_el = j_el.find("limit")
    if parent_el is None or child_el is None or origin_el is None or axis_el is None:
        raise ValueError(f"Joint {name!r}: missing parent, child, origin, or axis")
    if limit_el is None:
        lo, hi = -math.pi, math.pi
    else:
        lo = _f(limit_el.get("lower"), f"{name} limit.lower")
        hi = _f(limit_el.get("upper"), f"{name} limit.upper")
    return RevoluteJointParams(
        name=name,
        parent_link=cast(str, parent_el.get("link")),
        child_link=cast(str, child_el.get("link")),
        origin_xyz_m=_parse_vec3(origin_el.get("xyz"), f"{name} origin.xyz"),
        origin_rpy_rad=_parse_vec3(origin_el.get("rpy"), f"{name} origin.rpy"),
        axis=_norm_axis(_parse_vec3(axis_el.get("xyz"), f"{name} axis.xyz")),
        limit_lower_rad=lo,
        limit_upper_rad=hi,
    )


def _parse_fixed(j_el: ET.Element) -> FixedJointParams:
    name = cast(str, j_el.get("name"))
    parent_el = j_el.find("parent")
    child_el = j_el.find("child")
    origin_el = j_el.find("origin")
    if parent_el is None or child_el is None or origin_el is None:
        raise ValueError(f"Fixed joint {name!r}: missing parent, child, or origin")
    return FixedJointParams(
        name=name,
        parent_link=cast(str, parent_el.get("link")),
        child_link=cast(str, child_el.get("link")),
        origin_xyz_m=_parse_vec3(origin_el.get("xyz"), f"{name} origin.xyz"),
        origin_rpy_rad=_parse_vec3(origin_el.get("rpy"), f"{name} origin.rpy"),
    )


def _index_joints(root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}
    for j in root.findall("joint"):
        name = j.get("name")
        if name:
            out[name] = j
    return out


def _index_links(root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}
    for ln in root.findall("link"):
        name = ln.get("name")
        if name:
            out[name] = ln
    return out


def _satellite_from_base_link(link_el: ET.Element) -> SatelliteBodyParams:
    name = cast(str, link_el.get("name"))
    inertial = link_el.find("inertial")
    if inertial is None:
        raise ValueError(f"{name} missing <inertial>")
    mass_el = inertial.find("mass")
    inertia_el = inertial.find("inertia")
    origin_el = inertial.find("origin")
    if mass_el is None or inertia_el is None:
        raise ValueError(f"{name} inertial: need <mass> and <inertia>")
    com = (0.0, 0.0, 0.0)
    if origin_el is not None and origin_el.get("xyz"):
        com = _parse_vec3(origin_el.get("xyz"), f"{name} inertial.origin.xyz")
    length, radius = _first_cylinder_from_link(link_el)
    return SatelliteBodyParams(
        link_name=name,
        mass_kg=_f(mass_el.get("value"), "mass.value"),
        inertia=_parse_inertia(inertia_el),
        com_xyz_m=com,
        cylinder_radius_m=radius,
        cylinder_length_m=length,
    )


def _arm_definition(
    side: Literal["left", "right"],
    joints_by_name: Mapping[str, ET.Element],
    links_by_name: Mapping[str, ET.Element],
) -> RobotArmDefinition:
    prefix = f"{side}_"
    rjoints: List[RevoluteJointParams] = []
    for i in range(1, 5):
        jn = f"{prefix}joint{i}"
        j_el = joints_by_name.get(jn)
        if j_el is None or j_el.get("type") != "revolute":
            raise ValueError(f"Expected revolute joint {jn!r}")
        rjoints.append(_parse_revolute(j_el))
    ee_name = f"{prefix}ee_fixed"
    ee_el = joints_by_name.get(ee_name)
    if ee_el is None or ee_el.get("type") != "fixed":
        raise ValueError(f"Expected fixed joint {ee_name!r}")
    thruster_mount = _parse_fixed(ee_el)
    cylinders: List[ArmLinkCylinderParams] = []
    for i in range(1, 5):
        ln = f"{prefix}link{i}"
        link_el = links_by_name.get(ln)
        if link_el is None:
            raise ValueError(f"Missing link {ln!r}")
        length, radius = _first_cylinder_from_link(link_el)
        cylinders.append(
            ArmLinkCylinderParams(link_name=ln, length_m=length, radius_m=radius)
        )
    return RobotArmDefinition(
        side=side,
        joints=tuple(rjoints),
        thruster_mount=thruster_mount,
        link_cylinders=tuple(cylinders),
    )


def parse_robot_urdf(urdf_path: Path | str) -> ParsedRobotDescription:
    path = Path(urdf_path)
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("Root element must be <robot>")
    links = _index_links(root)
    joints = _index_joints(root)
    if BASE_LINK_NAME not in links:
        raise ValueError(f"URDF must define link {BASE_LINK_NAME!r}")
    satellite = _satellite_from_base_link(links[BASE_LINK_NAME])
    left_arm = _arm_definition("left", joints, links)
    right_arm = _arm_definition("right", joints, links)
    if left_arm.joints[0].parent_link != BASE_LINK_NAME:
        raise ValueError("left_joint1 must attach to base_link")
    if right_arm.joints[0].parent_link != BASE_LINK_NAME:
        raise ValueError("right_joint1 must attach to base_link")
    return ParsedRobotDescription(
        satellite=satellite,
        left_arm=left_arm,
        right_arm=right_arm,
    )


def default_urdf_path() -> Path:
    """Path to ``dual_arm_satellite.urdf`` next to the ``robot_description`` tree in ``src``."""
    return Path(__file__).resolve().parent.parent / "robot_description" / "urdf" / "dual_arm_satellite.urdf"
