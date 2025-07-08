"""
General mathematical utilities for rigid body dynamics

This module provides pure functional mathematical operations commonly used in
rigid body dynamics computations. All functions are pure (no side effects)
and work with immutable data.
"""

import numpy as np
from typing import Tuple
from ..types import Vector3, Matrix3x3, Scalar, validate_vector3, validate_matrix3x3


def create_skew_symmetric_matrix(vector: Vector3) -> Matrix3x3:
    """
    Pure function to create a skew-symmetric matrix from a 3D vector.

    Mathematical definition:
    For vector v = [x, y, z], creates matrix:
    [[ 0, -z,  y],
     [ z,  0, -x],
     [-y,  x,  0]]

    Property: skew_symmetric(v1) @ v2 = v1 × v2 (cross product)

    Args:
        vector: 3D vector [x, y, z]

    Returns:
        3x3 skew-symmetric matrix (new array, input unchanged)

    Raises:
        ValueError: If input is not exactly 3D
    """
    v = validate_vector3(vector)

    # Create new matrix without modifying input
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],   0.0]
    ])


# Alias for mathematical clarity
skew_symmetric = create_skew_symmetric_matrix
cross_product_matrix = create_skew_symmetric_matrix


def normalize_vector(vector: Vector3) -> Vector3:
    """
    Pure function to normalize a 3D vector to unit length.

    Mathematical operation: v_normalized = v / ||v||

    Args:
        vector: Input vector to normalize

    Returns:
        Unit vector in same direction (new array, input unchanged)
        Returns [1,0,0] if input is zero vector
    """
    v = validate_vector3(vector)
    norm = np.linalg.norm(v)

    # Handle zero vector case
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    return v / norm


def is_valid_rotation_matrix(matrix: Matrix3x3, tolerance: float = 1e-6) -> bool:
    """
    Pure function to check if matrix is a valid rotation matrix.

    Rotation matrix properties:
    1. Orthogonal: R @ R.T = I
    2. Proper: det(R) = +1 (not -1, which would be reflection)
    3. Shape: 3x3

    Args:
        matrix: Matrix to validate
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if matrix satisfies rotation matrix properties
    """
    try:
        R = validate_matrix3x3(matrix)
    except ValueError:
        return False

    # Check orthogonality: R @ R.T should equal identity
    orthogonality_test = R @ R.T
    identity = np.eye(3)
    is_orthogonal = np.allclose(orthogonality_test, identity, atol=tolerance)

    # Check proper rotation (determinant = +1)
    determinant = np.linalg.det(R)
    is_proper = np.isclose(determinant, 1.0, atol=tolerance)

    return is_orthogonal and is_proper


def rodrigues_formula(axis: Vector3, angle: float) -> Matrix3x3:
    """
    Pure function implementing Rodrigues' rotation formula.

    Mathematical formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    where [k]× is the skew-symmetric matrix of unit axis vector k

    Args:
        axis: 3D rotation axis (will be normalized to unit vector)
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix (new array)
    """
    # Normalize axis to unit vector
    unit_axis = normalize_vector(axis)

    # Create skew-symmetric matrix
    K = create_skew_symmetric_matrix(unit_axis)

    # Apply Rodrigues' formula
    identity = np.eye(3)
    sin_term = np.sin(angle) * K
    cos_term = (1 - np.cos(angle)) * (K @ K)

    return identity + sin_term + cos_term


def extract_angle_axis_from_rotation(rotation_matrix: Matrix3x3) -> Tuple[Vector3, float]:
    """
    Pure function to extract axis-angle representation from rotation matrix.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Tuple of (unit_axis, angle_radians)

    Raises:
        ValueError: If input is not a valid rotation matrix
    """
    R = np.asarray(rotation_matrix, dtype=float)

    if not is_valid_rotation_matrix(R):
        raise ValueError("Input is not a valid rotation matrix")

    # Extract angle from trace: trace(R) = 1 + 2*cos(θ)
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    if np.isclose(angle, 0):
        # Identity rotation
        return np.array([1.0, 0.0, 0.0]), 0.0

    elif np.isclose(angle, np.pi):
        # 180-degree rotation - special case
        # Find eigenvector with eigenvalue +1
        eigenvalues, eigenvectors = np.linalg.eig(R)
        axis_index = np.argmin(np.abs(eigenvalues - 1))
        axis = np.real(eigenvectors[:, axis_index])
        return normalize_vector(axis), angle

    else:
        # General case: extract axis from skew-symmetric part
        skew_part = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        axis = normalize_vector(skew_part)
        return axis, angle


# Convenient aliases for common usage
angle_axis_to_rotation_matrix = rodrigues_formula
rotation_matrix_to_angle_axis = extract_angle_axis_from_rotation
