"""
Mathematical utilities for rigid body dynamics

This module provides fundamental mathematical operations for 3D transformations,
rotations, and spatial algebra used in rigid body dynamics.

Key components:
- transform: 3D transformation matrices and operations
- rotation: Rotation representations and conversions
- spatial: Spatial vectors and matrices for rigid body dynamics
- utils: General mathematical utilities
"""

from .transform import Transform, SE3, compose_transforms, invert_transform
from .rotation import Rotation, SO3
from .spatial import SpatialVector, SpatialMatrix, SpatialInertia
from .utils import skew_symmetric, cross_product_matrix

__all__ = [
    "Transform",
    "SE3",
    "compose_transforms",
    "invert_transform",
    "Rotation",
    "SO3",
    "SpatialVector",
    "SpatialMatrix",
    "SpatialInertia",
    "skew_symmetric",
    "cross_product_matrix",
]
