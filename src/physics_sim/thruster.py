"""End-effector thruster: +Y of thruster frame, peak force (N), instantaneous linear impulse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from physics_sim.fk_chain import arm_joint_slice, compute_thrust_axis_world

if TYPE_CHECKING:
    from physics_sim.engine import PhysicsSimEngine


@dataclass
class Thruster:
    """
    Instantaneous reactionless thruster aligned with **+Y** of the URDF thruster link.

    When :meth:`fire` is called, a linear impulse ``J = throttle * max_thrust_n * impulse_window_s``
    (N·s) is applied at the satellite **center of mass** in the computed world direction
    (massless-arm / no moment from lever arm in this model).

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
        engine.apply_linear_impulse_world(J_world)
        return J_world
