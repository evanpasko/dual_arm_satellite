"""Load :class:`PhysicsSimEngine` from default URDF and report parsed mass / link data.

Run with output visible::

    uv run pytest tests/test_physics_engine_urdf_defaults.py -s

Or::

    PYTHONPATH=src pytest tests/test_physics_engine_urdf_defaults.py -s
"""

from __future__ import annotations

from physics_sim import PhysicsSimEngine


def _print_satellite(engine: PhysicsSimEngine) -> None:
    b = engine.satellite
    print("\n=== Satellite bus (URDF base_link) ===")
    print(f"  link_name:           {b.link_name}")
    print(f"  mass_kg:             {b.mass_kg}")
    print(f"  com_xyz_m:           {b.com_xyz_m}")
    print("  inertia (kg·m²):")
    print(f"    ixx={b.inertia.ixx}  ixy={b.inertia.ixy}  ixz={b.inertia.ixz}")
    print(f"    iyy={b.inertia.iyy}  iyz={b.inertia.iyz}  izz={b.inertia.izz}")
    print(f"  bus cylinder radius_m: {b.cylinder_radius_m}")
    print(f"  bus cylinder length_m: {b.cylinder_length_m}")


def _print_arm(engine: PhysicsSimEngine, side: str) -> None:
    arm = engine.left_arm if side == "left" else engine.right_arm
    assert arm.side == side
    print(f"\n=== {side.upper()} arm ===")
    print("  Revolute joints:")
    for j in arm.joints:
        print(f"    {j.name}")
        print(f"      parent -> child:  {j.parent_link} -> {j.child_link}")
        print(f"      origin_xyz_m:     {j.origin_xyz_m}")
        print(f"      origin_rpy_rad:   {j.origin_rpy_rad}")
        print(f"      axis (unit):      {j.axis}")
        print(
            f"      limits_rad:       [{j.limit_lower_rad}, {j.limit_upper_rad}]"
        )
    print("  Link cylinders (visual geometry):")
    for c in arm.link_cylinders:
        print(
            f"    {c.link_name}: length_m={c.length_m}, radius_m={c.radius_m}"
        )
    tm = arm.thruster_mount
    print("  Thruster mount (fixed):")
    print(f"    {tm.name}: {tm.parent_link} -> {tm.child_link}")
    print(f"    origin_xyz_m:   {tm.origin_xyz_m}")
    print(f"    origin_rpy_rad: {tm.origin_rpy_rad}")


def test_physics_engine_urdf_defaults_report() -> None:
    engine = PhysicsSimEngine.from_urdf()
    assert engine.urdf_path is not None
    print(f"\nURDF: {engine.urdf_path}")

    _print_satellite(engine)
    _print_arm(engine, "left")
    _print_arm(engine, "right")

    # Ground truth from current dual_arm_satellite.urdf (sanity checks)
    b = engine.satellite
    assert b.link_name == "base_link"
    assert b.mass_kg == 80.0
    assert b.inertia.ixx == 2.45 and b.inertia.iyy == 2.45 and b.inertia.izz == 4.9
    assert b.cylinder_radius_m == 0.35
    assert b.cylinder_length_m == 0.25

    for arm in (engine.left_arm, engine.right_arm):
        lengths = tuple(c.length_m for c in arm.link_cylinders)
        assert lengths == (0.15, 0.35, 0.30, 0.12)
        radii = {c.radius_m for c in arm.link_cylinders}
        assert radii == {0.03}

    print("\n(done — assertions passed)\n")
