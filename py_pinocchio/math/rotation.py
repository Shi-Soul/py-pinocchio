"""
3D rotation utilities using functional programming style

This module provides pure functions for working with 3D rotations in various
representations: rotation matrices, quaternions, Euler angles, and axis-angle.
All functions are pure (no side effects) and work with immutable data.
"""

import numpy as np
from typing import Tuple, NamedTuple
from .utils import (
    is_valid_rotation_matrix, 
    normalize_vector,
    rodrigues_formula,
    extract_angle_axis_from_rotation
)


class Quaternion(NamedTuple):
    """
    Immutable quaternion representation for rotations.
    
    Convention: q = w + xi + yj + zk where w is scalar part
    
    Fields:
        w: Scalar (real) part
        x, y, z: Vector (imaginary) parts
    """
    w: float
    x: float
    y: float
    z: float
    
    @property
    def vector_part(self) -> np.ndarray:
        """Get vector part [x, y, z] as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def scalar_part(self) -> float:
        """Get scalar part w."""
        return self.w


def create_identity_quaternion() -> Quaternion:
    """
    Create identity quaternion (no rotation).
    
    Returns:
        Identity quaternion: q = 1 + 0i + 0j + 0k
    """
    return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)


def create_quaternion(w: float, x: float, y: float, z: float) -> Quaternion:
    """
    Create normalized quaternion from components.
    
    Args:
        w, x, y, z: Quaternion components
        
    Returns:
        Normalized quaternion
    """
    # Normalize to unit quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return create_identity_quaternion()
    
    return Quaternion(w=w/norm, x=x/norm, y=y/norm, z=z/norm)


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> Quaternion:
    """
    Create quaternion from axis-angle representation.
    
    Mathematical formula: q = cos(θ/2) + sin(θ/2) * (xi + yj + zk)
    
    Args:
        axis: 3D rotation axis (will be normalized)
        angle: Rotation angle in radians
        
    Returns:
        Unit quaternion representing the rotation
    """
    unit_axis = normalize_vector(axis)
    half_angle = angle / 2
    
    w = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    
    return Quaternion(
        w=w,
        x=sin_half * unit_axis[0],
        y=sin_half * unit_axis[1], 
        z=sin_half * unit_axis[2]
    )


def quaternion_to_axis_angle(q: Quaternion) -> Tuple[np.ndarray, float]:
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        q: Unit quaternion
        
    Returns:
        Tuple of (axis, angle) where axis is unit vector and angle in radians
    """
    # Handle identity quaternion
    if abs(q.w) >= 1.0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    
    # Extract angle: w = cos(θ/2)
    angle = 2 * np.arccos(abs(q.w))
    
    # Extract axis: [x,y,z] = sin(θ/2) * axis
    sin_half_angle = np.sqrt(1 - q.w*q.w)
    if sin_half_angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    
    axis = q.vector_part / sin_half_angle
    
    # Ensure positive w for canonical representation
    if q.w < 0:
        axis = -axis
    
    return axis, angle


def quaternion_to_rotation_matrix(q: Quaternion) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Unit quaternion
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    
    # Rotation matrix from quaternion formula
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    """
    Convert rotation matrix to quaternion using Shepperd's method.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Unit quaternion
        
    Raises:
        ValueError: If R is not a valid rotation matrix
    """
    if not is_valid_rotation_matrix(R):
        raise ValueError("Invalid rotation matrix")
    
    # Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return Quaternion(w=w, x=x, y=y, z=z)


def multiply_quaternions(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Multiply two quaternions: result = q1 * q2
    
    Quaternion multiplication represents composition of rotations.
    
    Args:
        q1: First quaternion
        q2: Second quaternion
        
    Returns:
        Product quaternion (normalized)
    """
    w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    
    return create_quaternion(w, x, y, z)


def conjugate_quaternion(q: Quaternion) -> Quaternion:
    """
    Compute quaternion conjugate: q* = w - xi - yj - zk
    
    For unit quaternions, conjugate equals inverse.
    
    Args:
        q: Input quaternion
        
    Returns:
        Conjugate quaternion
    """
    return Quaternion(w=q.w, x=-q.x, y=-q.y, z=-q.z)


def euler_angles_to_rotation_matrix(roll: float, pitch: float, yaw: float, 
                                  convention: str = 'xyz') -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        convention: Rotation order ('xyz', 'zyx', etc.)
        
    Returns:
        3x3 rotation matrix
    """
    if convention != 'xyz':
        raise NotImplementedError(f"Convention '{convention}' not implemented")
    
    # Individual rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Compose rotations: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


# Convenient aliases
SO3 = np.ndarray  # Type alias for rotation matrices
Rotation = np.ndarray  # Type alias for rotation matrices
