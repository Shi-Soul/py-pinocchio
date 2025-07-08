"""
3D transformation utilities using functional programming style

This module provides pure functions for working with 3D transformations,
including homogeneous transformation matrices and SE(3) operations.
All functions are pure (no side effects) and work with immutable data.
"""

import numpy as np
from typing import NamedTuple
from .utils import is_valid_rotation_matrix
from ..types import Vector3, Matrix3x3, Matrix4x4, validate_vector3, validate_matrix3x3


class Transform(NamedTuple):
    """
    Immutable 3D transformation representation.

    Represents a rigid body transformation with rotation and translation.
    Uses named tuple for immutability and clear field access.

    Fields:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    """
    rotation: Matrix3x3
    translation: Vector3

    def __post_init__(self):
        """Validate the transformation components."""
        if not is_valid_rotation_matrix(self.rotation):
            raise ValueError("Invalid rotation matrix")
        try:
            validate_vector3(self.translation)
        except ValueError:
            raise ValueError("Translation must be 3D vector")


def create_identity_transform() -> Transform:
    """
    Create identity transformation (no rotation, no translation).
    
    Returns:
        Identity transform: R = I, t = 0
    """
    return Transform(
        rotation=np.eye(3),
        translation=np.zeros(3)
    )


def create_transform(rotation: Matrix3x3, translation: Vector3) -> Transform:
    """
    Create a transformation from rotation matrix and translation vector.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector

    Returns:
        Transform object (immutable)

    Raises:
        ValueError: If rotation or translation are invalid
    """
    R = validate_matrix3x3(rotation)
    t = validate_vector3(translation)

    if not is_valid_rotation_matrix(R):
        raise ValueError("Invalid rotation matrix")

    return Transform(rotation=R.copy(), translation=t.copy())


def create_translation_transform(translation: np.ndarray) -> Transform:
    """
    Create pure translation transformation (no rotation).
    
    Args:
        translation: 3D translation vector
        
    Returns:
        Transform with identity rotation and given translation
    """
    return create_transform(
        rotation=np.eye(3),
        translation=translation
    )


def create_rotation_transform(rotation: np.ndarray) -> Transform:
    """
    Create pure rotation transformation (no translation).
    
    Args:
        rotation: 3x3 rotation matrix
        
    Returns:
        Transform with given rotation and zero translation
    """
    return create_transform(
        rotation=rotation,
        translation=np.zeros(3)
    )


def transform_to_homogeneous_matrix(transform: Transform) -> np.ndarray:
    """
    Convert Transform to 4x4 homogeneous transformation matrix.
    
    Matrix format:
    [R  t]
    [0  1]
    
    Args:
        transform: Transform object
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    H = np.eye(4)
    H[:3, :3] = transform.rotation
    H[:3, 3] = transform.translation
    return H


def homogeneous_matrix_to_transform(matrix: np.ndarray) -> Transform:
    """
    Convert 4x4 homogeneous matrix to Transform object.
    
    Args:
        matrix: 4x4 homogeneous transformation matrix
        
    Returns:
        Transform object
        
    Raises:
        ValueError: If matrix is not valid homogeneous transformation
    """
    H = np.asarray(matrix, dtype=float)
    
    if H.shape != (4, 4):
        raise ValueError("Matrix must be 4x4")
    
    # Check bottom row is [0, 0, 0, 1]
    expected_bottom = np.array([0, 0, 0, 1])
    if not np.allclose(H[3, :], expected_bottom):
        raise ValueError("Invalid homogeneous matrix bottom row")
    
    rotation = H[:3, :3]
    translation = H[:3, 3]
    
    return create_transform(rotation, translation)


def compose_transforms(transform1: Transform, transform2: Transform) -> Transform:
    """
    Compose two transformations: result = transform1 ∘ transform2
    
    Mathematical operation: T_result = T1 * T2
    Point transformation: p' = T1(T2(p))
    
    Args:
        transform1: First transformation (applied second)
        transform2: Second transformation (applied first)
        
    Returns:
        Composed transformation (new Transform object)
    """
    # Compose rotations: R_result = R1 @ R2
    composed_rotation = transform1.rotation @ transform2.rotation
    
    # Compose translations: t_result = R1 @ t2 + t1
    composed_translation = transform1.rotation @ transform2.translation + transform1.translation
    
    return Transform(
        rotation=composed_rotation,
        translation=composed_translation
    )


def invert_transform(transform: Transform) -> Transform:
    """
    Compute inverse of transformation.
    
    Mathematical operation: T^(-1) where T * T^(-1) = I
    
    For transform T = (R, t):
    T^(-1) = (R^T, -R^T * t)
    
    Args:
        transform: Transform to invert
        
    Returns:
        Inverse transformation (new Transform object)
    """
    # Inverse rotation: R^(-1) = R^T (for rotation matrices)
    inverse_rotation = transform.rotation.T
    
    # Inverse translation: t^(-1) = -R^T * t
    inverse_translation = -inverse_rotation @ transform.translation
    
    return Transform(
        rotation=inverse_rotation,
        translation=inverse_translation
    )


def transform_point(transform: Transform, point: np.ndarray) -> np.ndarray:
    """
    Apply transformation to a 3D point.
    
    Mathematical operation: p' = R * p + t
    
    Args:
        transform: Transformation to apply
        point: 3D point to transform
        
    Returns:
        Transformed point (new array)
    """
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        raise ValueError("Point must be 3D vector")
    
    return transform.rotation @ p + transform.translation


def transform_vector(transform: Transform, vector: np.ndarray) -> np.ndarray:
    """
    Apply only rotation part of transformation to a vector.
    
    Vectors represent directions, so translation doesn't apply.
    Mathematical operation: v' = R * v
    
    Args:
        transform: Transformation (only rotation used)
        vector: 3D vector to transform
        
    Returns:
        Transformed vector (new array)
    """
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError("Vector must be 3D")
    
    return transform.rotation @ v


# Convenient alias for SE(3) operations
SE3 = Transform
