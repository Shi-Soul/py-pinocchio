"""
Type definitions for py-pinocchio

This module provides consistent type annotations throughout the library,
especially for numpy arrays and common data structures.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any, TypeAlias

# For older Python versions that don't have numpy.typing
try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray

# Basic numpy array types
Vector3: TypeAlias = np.ndarray  # 3D vector
Vector6: TypeAlias = np.ndarray  # 6D spatial vector
Matrix3x3: TypeAlias = np.ndarray  # 3x3 matrix (rotation, inertia)
Matrix4x4: TypeAlias = np.ndarray  # 4x4 homogeneous transformation
Matrix6x6: TypeAlias = np.ndarray  # 6x6 spatial matrix
MatrixNxN: TypeAlias = np.ndarray  # NxN matrix (mass matrix, Jacobian)

# Joint configuration types
JointPosition: TypeAlias = float
JointVelocity: TypeAlias = float
JointAcceleration: TypeAlias = float
JointTorque: TypeAlias = float

JointPositions: TypeAlias = np.ndarray  # Array of joint positions
JointVelocities: TypeAlias = np.ndarray  # Array of joint velocities
JointAccelerations: TypeAlias = np.ndarray  # Array of joint accelerations
JointTorques: TypeAlias = np.ndarray  # Array of joint torques

# Quaternion type (w, x, y, z)
Quaternion: TypeAlias = np.ndarray

# Time type
Time: TypeAlias = float

# Generic numeric types
Scalar: TypeAlias = Union[int, float]
Array: TypeAlias = np.ndarray

# String identifiers
LinkName: TypeAlias = str
JointName: TypeAlias = str
FrameName: TypeAlias = str

# Configuration validation
def validate_vector3(v: Any) -> Vector3:
    """Validate and convert to 3D vector."""
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {arr.shape}")
    return arr

def validate_matrix3x3(m: Any) -> Matrix3x3:
    """Validate and convert to 3x3 matrix."""
    arr = np.asarray(m, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {arr.shape}")
    return arr

def validate_joint_positions(q: Any, expected_dof: int) -> JointPositions:
    """Validate joint positions array."""
    arr = np.asarray(q, dtype=float)
    if arr.shape != (expected_dof,):
        raise ValueError(f"Expected {expected_dof} joint positions, got {arr.shape}")
    return arr

def validate_quaternion(quat: Any) -> Quaternion:
    """Validate and normalize quaternion."""
    arr = np.asarray(quat, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Expected quaternion [w,x,y,z], got shape {arr.shape}")
    
    # Normalize
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return arr / norm
