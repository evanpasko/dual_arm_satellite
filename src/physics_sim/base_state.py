"""Satellite ``base_link`` pose, optional joint seed, and time-series logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Quaternion order matches ROS ``geometry_msgs/Quaternion``: x, y, z, w (w is scalar part).


def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion; ``q`` is (x, y, z, w)."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-15:
        raise ValueError("Quaternion norm is zero")
    return q / n


@dataclass
class BaseLinkPose:
    """
    Cartesian pose of the satellite bus in the world / inertial frame.

    - ``position_m``: (3,) translation in metres.
    - ``quaternion_xyzw``: (4,) unit quaternion; rotation maps vectors from
      **body (base_link) frame → world frame** (same sense as applying q to a body-fixed vector).
    """

    position_m: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        self.position_m = np.asarray(self.position_m, dtype=float).reshape(3)
        self.quaternion_xyzw = normalize_quaternion_xyzw(self.quaternion_xyzw)

    @staticmethod
    def identity() -> BaseLinkPose:
        return BaseLinkPose(
            np.zeros(3, dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )

    def copy(self) -> BaseLinkPose:
        return BaseLinkPose(self.position_m.copy(), self.quaternion_xyzw.copy())


@dataclass
class PoseTimeSeries:
    """Samples of ``base_link`` pose over simulation time."""

    times_s: List[float] = field(default_factory=list)
    positions_m: List[np.ndarray] = field(default_factory=list)
    quaternions_xyzw: List[np.ndarray] = field(default_factory=list)

    def append(self, time_s: float, pose: BaseLinkPose) -> None:
        self.times_s.append(float(time_s))
        self.positions_m.append(pose.position_m.copy())
        self.quaternions_xyzw.append(pose.quaternion_xyzw.copy())

    def clear(self) -> None:
        self.times_s.clear()
        self.positions_m.clear()
        self.quaternions_xyzw.clear()

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times_s, positions_m (N,3), quaternions_xyzw (N,4))``."""
        if not self.times_s:
            return (
                np.zeros(0, dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 4), dtype=float),
            )
        t = np.asarray(self.times_s, dtype=float)
        p = np.stack(self.positions_m, axis=0)
        q = np.stack(self.quaternions_xyzw, axis=0)
        return t, p, q

    def __len__(self) -> int:
        return len(self.times_s)


def make_initial_joint_positions(
    left_arm_rad: Optional[np.ndarray] = None,
    right_arm_rad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build an 8-vector ``[left j1..j4, right j1..j4]`` in radians.

    Omitted arms default to zeros.
    """
    left = np.zeros(4, dtype=float) if left_arm_rad is None else np.asarray(left_arm_rad, dtype=float).reshape(4)
    right = np.zeros(4, dtype=float) if right_arm_rad is None else np.asarray(right_arm_rad, dtype=float).reshape(4)
    return np.concatenate([left, right])
