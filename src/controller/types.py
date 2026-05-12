"""Shared controller value-types: state error and control command.

These are intended as the I/O contract between any controller subclass and the
simulation driver. Putting them at the package level (rather than inside a
specific controller file) means bang-bang, PID, LQR, etc. can all be swapped
through the same interface.

Frames are baked into field names. The simulation stores translational state in
the world / inertial frame and angular state in the body (``base_link``) frame,
and these dataclasses follow the same convention so callers don't have to guess
which transform to apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _frozen_array(value: np.ndarray, shape: int, name: str) -> np.ndarray:
    """Coerce to ``float64`` of the expected length and mark read-only.

    A defensive copy is taken so external mutation of the caller's array can't
    silently alter this frozen dataclass.
    """
    arr = np.asarray(value, dtype=float).reshape(shape).copy()
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class StateError:
    """Difference between current and target satellite state.

    ``position_error_world_m`` is ``p_target - p_current`` in the world /
    inertial frame.

    ``rotation_error_body_rad`` is a small-angle rotation vector (axis times
    angle) in the body frame: applying it to the current ``base_link``
    orientation yields the target orientation. The body frame is used because
    :attr:`physics_sim.PhysicsSimEngine.angular_velocity_body_rad_s` lives in
    the body frame, so the rotation error and angular-velocity error compose
    consistently without re-rotation.

    Velocity errors are ``target - current`` in their named frames. They are
    optional in practice: a bang-bang controller usually ignores them, while a
    PID needs them as the derivative term.

    All four arrays are shape ``(3,)`` and immutable after construction.
    """

    position_error_world_m: np.ndarray
    rotation_error_body_rad: np.ndarray
    linear_velocity_error_world_m_s: np.ndarray
    angular_velocity_error_body_rad_s: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "position_error_world_m",
            "rotation_error_body_rad",
            "linear_velocity_error_world_m_s",
            "angular_velocity_error_body_rad_s",
        ):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), 3, name))

    @classmethod
    def zero(cls) -> StateError:
        """Build a zero-error instance (useful as a tolerance pass-through)."""
        zeros = lambda: np.zeros(3, dtype=float)
        return cls(zeros(), zeros(), zeros(), zeros())

    @property
    def position_error_norm_m(self) -> float:
        return float(np.linalg.norm(self.position_error_world_m))

    @property
    def rotation_error_norm_rad(self) -> float:
        return float(np.linalg.norm(self.rotation_error_body_rad))

    @property
    def linear_velocity_error_norm_m_s(self) -> float:
        return float(np.linalg.norm(self.linear_velocity_error_world_m_s))

    @property
    def angular_velocity_error_norm_rad_s(self) -> float:
        return float(np.linalg.norm(self.angular_velocity_error_body_rad_s))

    def is_within_tolerance(
        self,
        *,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        linear_velocity_tolerance_m_s: float = float("inf"),
        angular_velocity_tolerance_rad_s: float = float("inf"),
    ) -> bool:
        """All error norms below the supplied tolerances.

        Velocity tolerances default to ``inf`` so a position+rotation-only
        deadband works without passing velocity arguments.
        """
        return (
            self.position_error_norm_m < position_tolerance_m
            and self.rotation_error_norm_rad < rotation_tolerance_rad
            and self.linear_velocity_error_norm_m_s < linear_velocity_tolerance_m_s
            and self.angular_velocity_error_norm_rad_s < angular_velocity_tolerance_rad_s
        )


@dataclass(frozen=True)
class ControlCommand:
    """Actuation request produced by one ``calculate_control`` tick.

    - ``joint_positions_rad`` is the 8-vector ``[left j1..4, right j1..4]``,
      matching :attr:`physics_sim.PhysicsSimEngine.joint_positions_rad`. Under
      the "joints teleport" simplifying assumption, the engine snaps to this
      every tick.
    - ``left_throttle`` and ``right_throttle`` are in ``[0, 1]``. Per
      :class:`physics_sim.Thruster`, the applied impulse magnitude is
      ``throttle * max_thrust_n * impulse_window_s``. Bang-bang clamps to
      ``{0, 1}``; PID / LQR use the full interval.
    """

    joint_positions_rad: np.ndarray
    left_throttle: float
    right_throttle: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_positions_rad",
            _frozen_array(self.joint_positions_rad, 8, "joint_positions_rad"),
        )
        for name in ("left_throttle", "right_throttle"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
            object.__setattr__(self, name, value)

    @classmethod
    def hold(
        cls,
        joint_positions_rad: np.ndarray,
    ) -> ControlCommand:
        """Convenience constructor: keep current joints, fire no thrusters."""
        return cls(joint_positions_rad=joint_positions_rad, left_throttle=0.0, right_throttle=0.0)

    @property
    def left_joint_positions_rad(self) -> np.ndarray:
        """Slice ``[j1..j4]`` of the left arm. Shares memory; do not modify."""
        return self.joint_positions_rad[0:4]

    @property
    def right_joint_positions_rad(self) -> np.ndarray:
        """Slice ``[j1..j4]`` of the right arm. Shares memory; do not modify."""
        return self.joint_positions_rad[4:8]

    @property
    def is_left_firing(self) -> bool:
        return self.left_throttle > 0.0

    @property
    def is_right_firing(self) -> bool:
        return self.right_throttle > 0.0
