"""Build 3×3 inertia tensor from URDF-style diagonal / symmetric components."""

from __future__ import annotations

import numpy as np

from robot_description.models import Inertia, SatelliteBodyParams


def inertia_tensor_body_kg_m2(I: Inertia) -> np.ndarray:
    """Symmetric inertia matrix in the body / ``base_link`` frame (kg·m²)."""
    return np.array(
        [
            [I.ixx, I.ixy, I.ixz],
            [I.ixy, I.iyy, I.iyz],
            [I.ixz, I.iyz, I.izz],
        ],
        dtype=float,
    )


def inertia_tensor_body_from_satellite(sat: SatelliteBodyParams) -> np.ndarray:
    return inertia_tensor_body_kg_m2(sat.inertia)


def invert_inertia_3x3(I: np.ndarray) -> np.ndarray:
    """Inverse of SPD inertia; falls back to ``numpy.linalg.pinv`` if singular."""
    I = np.asarray(I, dtype=float).reshape(3, 3)
    try:
        return np.linalg.inv(I)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(I)
