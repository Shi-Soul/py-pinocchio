"""
Spatial algebra for rigid body dynamics using functional programming style

This module implements spatial vectors and matrices used in rigid body dynamics.
Spatial algebra provides an elegant mathematical framework for representing
forces, velocities, and inertias in 6D space.

All functions are pure (no side effects) and work with immutable data.
"""

import numpy as np
from typing import NamedTuple, Union
from .utils import create_skew_symmetric_matrix
from .transform import Transform


class SpatialVector(NamedTuple):
    """
    Immutable 6D spatial vector representation.
    
    A spatial vector combines angular and linear components:
    - For motion: [angular_velocity, linear_velocity]
    - For force: [moment, force]
    
    Fields:
        angular: 3D angular component (top 3 elements)
        linear: 3D linear component (bottom 3 elements)
    """
    angular: np.ndarray
    linear: np.ndarray
    
    def __post_init__(self):
        """Validate spatial vector components."""
        if np.asarray(self.angular).shape != (3,):
            raise ValueError("Angular component must be 3D vector")
        if np.asarray(self.linear).shape != (3,):
            raise ValueError("Linear component must be 3D vector")
    
    @property
    def vector(self) -> np.ndarray:
        """Get as 6D numpy array [angular; linear]."""
        return np.concatenate([self.angular, self.linear])


def create_spatial_vector(angular: np.ndarray, linear: np.ndarray) -> SpatialVector:
    """
    Create spatial vector from angular and linear components.
    
    Args:
        angular: 3D angular component
        linear: 3D linear component
        
    Returns:
        SpatialVector object (immutable)
    """
    return SpatialVector(
        angular=np.asarray(angular, dtype=float).copy(),
        linear=np.asarray(linear, dtype=float).copy()
    )


def create_zero_spatial_vector() -> SpatialVector:
    """
    Create zero spatial vector (no motion/force).
    
    Returns:
        Zero spatial vector
    """
    return create_spatial_vector(
        angular=np.zeros(3),
        linear=np.zeros(3)
    )


def create_spatial_motion(angular_velocity: np.ndarray, linear_velocity: np.ndarray) -> SpatialVector:
    """
    Create spatial motion vector (twist).
    
    Args:
        angular_velocity: 3D angular velocity
        linear_velocity: 3D linear velocity
        
    Returns:
        Spatial motion vector
    """
    return create_spatial_vector(angular_velocity, linear_velocity)


def create_spatial_force(moment: np.ndarray, force: np.ndarray) -> SpatialVector:
    """
    Create spatial force vector (wrench).
    
    Args:
        moment: 3D moment vector
        force: 3D force vector
        
    Returns:
        Spatial force vector
    """
    return create_spatial_vector(moment, force)


def add_spatial_vectors(sv1: SpatialVector, sv2: SpatialVector) -> SpatialVector:
    """
    Add two spatial vectors component-wise.
    
    Args:
        sv1, sv2: Spatial vectors to add
        
    Returns:
        Sum of spatial vectors
    """
    return create_spatial_vector(
        angular=sv1.angular + sv2.angular,
        linear=sv1.linear + sv2.linear
    )


def scale_spatial_vector(sv: SpatialVector, scalar: float) -> SpatialVector:
    """
    Scale spatial vector by scalar.
    
    Args:
        sv: Spatial vector to scale
        scalar: Scaling factor
        
    Returns:
        Scaled spatial vector
    """
    return create_spatial_vector(
        angular=scalar * sv.angular,
        linear=scalar * sv.linear
    )


def spatial_cross_product(motion: SpatialVector, vector: SpatialVector) -> SpatialVector:
    """
    Compute spatial cross product: motion × vector
    
    This operation is fundamental in rigid body dynamics for:
    - Computing Coriolis and centrifugal effects
    - Transforming spatial vectors between frames
    
    Mathematical definition:
    [ω; v] × [a; b] = [ω × a; ω × b + v × a]
    
    Args:
        motion: Spatial motion vector (first operand)
        vector: Spatial vector (second operand)
        
    Returns:
        Cross product result
    """
    omega = motion.angular
    v = motion.linear
    a = vector.angular
    b = vector.linear
    
    # Cross product formula for spatial vectors
    result_angular = np.cross(omega, a)
    result_linear = np.cross(omega, b) + np.cross(v, a)
    
    return create_spatial_vector(result_angular, result_linear)


def spatial_cross_product_dual(force: SpatialVector, motion: SpatialVector) -> SpatialVector:
    """
    Compute dual spatial cross product: force ×* motion
    
    This is the dual operation used for force transformations.
    
    Mathematical definition:
    [n; f] ×* [ω; v] = [n × ω + f × v; f × ω]
    
    Args:
        force: Spatial force vector
        motion: Spatial motion vector
        
    Returns:
        Dual cross product result
    """
    n = force.angular  # moment
    f = force.linear   # force
    omega = motion.angular
    v = motion.linear
    
    # Dual cross product formula
    result_angular = np.cross(n, omega) + np.cross(f, v)
    result_linear = np.cross(f, omega)
    
    return create_spatial_vector(result_angular, result_linear)


class SpatialMatrix(NamedTuple):
    """
    Immutable 6x6 spatial matrix representation.
    
    Spatial matrices are used for:
    - Spatial inertia matrices
    - Spatial transformation matrices
    - Articulated body inertias
    
    Fields:
        matrix: 6x6 numpy array
    """
    matrix: np.ndarray
    
    def __post_init__(self):
        """Validate spatial matrix."""
        if np.asarray(self.matrix).shape != (6, 6):
            raise ValueError("Spatial matrix must be 6x6")


def create_spatial_matrix(matrix: np.ndarray) -> SpatialMatrix:
    """
    Create spatial matrix from 6x6 array.
    
    Args:
        matrix: 6x6 numpy array
        
    Returns:
        SpatialMatrix object (immutable)
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (6, 6):
        raise ValueError("Matrix must be 6x6")
    
    return SpatialMatrix(matrix=m.copy())


def create_zero_spatial_matrix() -> SpatialMatrix:
    """
    Create 6x6 zero spatial matrix.
    
    Returns:
        Zero spatial matrix
    """
    return create_spatial_matrix(np.zeros((6, 6)))


def create_identity_spatial_matrix() -> SpatialMatrix:
    """
    Create 6x6 identity spatial matrix.
    
    Returns:
        Identity spatial matrix
    """
    return create_spatial_matrix(np.eye(6))


def multiply_spatial_matrix_vector(matrix: SpatialMatrix, vector: SpatialVector) -> SpatialVector:
    """
    Multiply spatial matrix by spatial vector.
    
    Args:
        matrix: 6x6 spatial matrix
        vector: 6D spatial vector
        
    Returns:
        Result spatial vector
    """
    result_6d = matrix.matrix @ vector.vector
    return create_spatial_vector(
        angular=result_6d[:3],
        linear=result_6d[3:]
    )


def create_spatial_transformation_matrix(transform: Transform) -> SpatialMatrix:
    """
    Create 6x6 spatial transformation matrix from SE(3) transform.
    
    The spatial transformation matrix has the form:
    [R   0  ]
    [tR  R  ]
    
    where R is rotation matrix and t is translation as skew-symmetric matrix.
    
    Args:
        transform: SE(3) transformation
        
    Returns:
        6x6 spatial transformation matrix
    """
    R = transform.rotation
    t_skew = create_skew_symmetric_matrix(transform.translation)
    
    # Build 6x6 transformation matrix
    spatial_transform = np.zeros((6, 6))
    spatial_transform[:3, :3] = R
    spatial_transform[3:, :3] = t_skew @ R
    spatial_transform[3:, 3:] = R
    
    return create_spatial_matrix(spatial_transform)


def create_spatial_inertia_matrix(mass: float, center_of_mass: np.ndarray, 
                                inertia_tensor: np.ndarray) -> SpatialMatrix:
    """
    Create spatial inertia matrix from physical parameters.
    
    The spatial inertia matrix has the form:
    [I + m*c×c×   m*c×]
    [m*c×         m*I ]
    
    where I is inertia tensor, m is mass, c is center of mass, and c× is skew-symmetric.
    
    Args:
        mass: Body mass
        center_of_mass: 3D center of mass position
        inertia_tensor: 3x3 inertia tensor about center of mass
        
    Returns:
        6x6 spatial inertia matrix
    """
    m = float(mass)
    c = np.asarray(center_of_mass, dtype=float)
    I = np.asarray(inertia_tensor, dtype=float)
    
    if c.shape != (3,):
        raise ValueError("Center of mass must be 3D vector")
    if I.shape != (3, 3):
        raise ValueError("Inertia tensor must be 3x3 matrix")
    
    # Create skew-symmetric matrix of center of mass
    c_skew = create_skew_symmetric_matrix(c)
    
    # Build spatial inertia matrix
    spatial_inertia = np.zeros((6, 6))
    
    # Upper left: I + m*c×c×
    spatial_inertia[:3, :3] = I + m * (c_skew @ c_skew)
    
    # Upper right: m*c×
    spatial_inertia[:3, 3:] = m * c_skew
    
    # Lower left: m*c×
    spatial_inertia[3:, :3] = m * c_skew
    
    # Lower right: m*I
    spatial_inertia[3:, 3:] = m * np.eye(3)
    
    return create_spatial_matrix(spatial_inertia)


# Type alias for clarity
SpatialInertia = SpatialMatrix
