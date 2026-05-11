"""End-effector thruster: +Y of thruster frame, peak force (N), instantaneous linear impulse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from physics_sim.fk_chain import (
    arm_joint_slice,
    compute_thrust_axis_world,
    thruster_application_point_world_m,
)

if TYPE_CHECKING:
    from physics_sim.engine import PhysicsSimEngine


@dataclass
class Thruster:
    """
    Instantaneous thruster aligned with **+Y** of the URDF thruster link.

    When :meth:`fire` is called, a linear impulse ``J = throttle * max_thrust_n * impulse_window_s``
    (N·s) is applied along the thrust line through the **thruster frame origin** (URDF FK).
    The engine updates **linear** momentum at the CoM (``Δv = J / m``) and **angular**
    momentum about the CoM via ``Δω = I^{-1} (r_b × J_b)`` with lever arm ``r`` from CoM
    to the application point (``base_link`` inertial origin is the CoM in the URDF).

    Parameters
    ----------
    max_thrust_n
        Peak thrust when ``throttle == 1`` (newtons).
    impulse_window_s
        Effective duration used to convert force to impulse per burn (seconds). Tune to match
        your discrete-time step or notional burn length.
    """

    side: Literal["left", "right"]
    max_thrust_n: float
    impulse_window_s: float = 0.01

    def __post_init__(self) -> None:
        if self.max_thrust_n < 0.0:
            raise ValueError("max_thrust_n must be non-negative")
        if self.impulse_window_s <= 0.0:
            raise ValueError("impulse_window_s must be positive")

    @staticmethod
    def thrust_axis_thruster_frame() -> np.ndarray:
        """Unit vector along thrust (+Y of ``*_ee_thruster`` link)."""
        return np.array([0.0, 1.0, 0.0], dtype=float)

    def thrust_direction_world(self, engine: PhysicsSimEngine) -> np.ndarray:
        arm = engine.left_arm if self.side == "left" else engine.right_arm
        q = arm_joint_slice(engine.joint_positions_rad, self.side)
        return compute_thrust_axis_world(
            engine.base_pose,
            arm,
            q,
            thrust_axis_thruster=self.thrust_axis_thruster_frame(),
        )

    def fire(
        self,
        engine: PhysicsSimEngine,
        throttle: float,
        *,
        impulse_window_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply one instantaneous linear impulse in world frame (N·s).

        Returns the impulse vector applied (world frame).
        """
        dt = self.impulse_window_s if impulse_window_s is None else float(impulse_window_s)
        if dt <= 0.0:
            raise ValueError("impulse_window_s must be positive")
        t = float(np.clip(throttle, 0.0, 1.0))
        J_mag = t * self.max_thrust_n * dt
        direction = self.thrust_direction_world(engine)
        J_world = J_mag * direction
        arm = engine.left_arm if self.side == "left" else engine.right_arm
        q = arm_joint_slice(engine.joint_positions_rad, self.side)
        p_app = thruster_application_point_world_m(engine.base_pose, arm, q)
        engine.apply_impulse_world(J_world, application_point_world_m=p_app)
        return J_world
