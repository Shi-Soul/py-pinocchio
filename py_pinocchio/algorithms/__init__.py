"""
Rigid body dynamics algorithms using functional programming style

This module provides pure functional implementations of core robotics algorithms:
- Forward kinematics: Computing link positions from joint angles
- Inverse dynamics: Computing joint torques from motion
- Forward dynamics: Computing joint accelerations from torques
- Jacobian computation: Computing velocity relationships

All algorithms are implemented as pure functions for educational clarity.
"""

from .kinematics import (
    ForwardKinematics,
    compute_forward_kinematics,
    compute_link_transforms,
    get_link_transform,
    get_link_position,
    get_link_orientation
)

from .jacobian import (
    JacobianComputer,
    compute_geometric_jacobian,
    compute_analytical_jacobian,
    compute_jacobian_time_derivative
)

from .dynamics import (
    InverseDynamics,
    ForwardDynamics,
    compute_inverse_dynamics,
    compute_forward_dynamics,
    compute_mass_matrix,
    compute_coriolis_forces,
    compute_coriolis_matrix,
    compute_gravity_vector
)

__all__ = [
    # Forward kinematics
    "ForwardKinematics",
    "compute_forward_kinematics",
    "compute_link_transforms",
    "get_link_transform",
    "get_link_position", 
    "get_link_orientation",
    
    # Jacobian computation
    "JacobianComputer",
    "compute_geometric_jacobian",
    "compute_analytical_jacobian",
    "compute_jacobian_time_derivative",
    
    # Dynamics
    "InverseDynamics",
    "ForwardDynamics",
    "compute_inverse_dynamics",
    "compute_forward_dynamics",
    "compute_mass_matrix",
    "compute_coriolis_forces",
    "compute_coriolis_matrix",
    "compute_gravity_vector",
]
