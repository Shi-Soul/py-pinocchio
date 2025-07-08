"""
py-pinocchio: A fast and flexible implementation of Rigid Body Dynamics algorithms

This package provides educational implementations of rigid body dynamics algorithms
commonly used in robotics, including forward/inverse kinematics, dynamics, and
Jacobian computations.

Main classes:
- RobotModel: Main interface for robot manipulation
- Joint: Represents a robot joint
- Link: Represents a robot link
- Transform: 3D transformation utilities

Main modules:
- math: Mathematical utilities for 3D transformations
- parsers: URDF and MJCF file parsers
- algorithms: Core dynamics algorithms
- visualization: Visualization tools
"""

__version__ = "0.1.0"
__author__ = "Educational Implementation"

# Import main classes for easy access
from .model import RobotModel, Joint, Link, JointType, create_robot_model
from .math.transform import Transform, SE3, create_transform, create_identity_transform
from .math.spatial import SpatialVector, SpatialMatrix, create_spatial_vector
from .math.rotation import Quaternion, euler_angles_to_rotation_matrix

# Import parsers
from .parsers.urdf_parser import URDFParser, parse_urdf_file, parse_urdf_string
from .parsers.mjcf_parser import MJCFParser, parse_mjcf_file, parse_mjcf_string

# Import algorithms
from .algorithms.kinematics import (
    ForwardKinematics, compute_forward_kinematics,
    get_link_transform, get_link_position, get_link_orientation
)
from .algorithms.dynamics import (
    InverseDynamics, ForwardDynamics, compute_inverse_dynamics, compute_forward_dynamics,
    compute_mass_matrix, compute_coriolis_forces, compute_gravity_vector
)
from .algorithms.jacobian import (
    JacobianComputer, compute_geometric_jacobian, compute_analytical_jacobian
)

# Import visualization (optional - requires matplotlib)
try:
    from .visualization import (
        RobotVisualizer, plot_robot_2d, plot_robot_3d,
        animate_robot_motion, plot_workspace, plot_jacobian_analysis
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

__all__ = [
    # Core classes
    "RobotModel",
    "Joint",
    "Link",
    "JointType",
    "create_robot_model",
    "Transform",
    "SE3",
    "create_transform",
    "create_identity_transform",
    "SpatialVector",
    "SpatialMatrix",
    "create_spatial_vector",
    "Quaternion",
    "euler_angles_to_rotation_matrix",

    # Parsers
    "URDFParser",
    "parse_urdf_file",
    "parse_urdf_string",
    "MJCFParser",
    "parse_mjcf_file",
    "parse_mjcf_string",

    # Algorithms
    "ForwardKinematics",
    "compute_forward_kinematics",
    "get_link_transform",
    "get_link_position",
    "get_link_orientation",
    "InverseDynamics",
    "ForwardDynamics",
    "compute_inverse_dynamics",
    "compute_forward_dynamics",
    "compute_mass_matrix",
    "compute_coriolis_forces",
    "compute_gravity_vector",
    "JacobianComputer",
    "compute_geometric_jacobian",
    "compute_analytical_jacobian",
]

# Add visualization to __all__ if available
if _HAS_VISUALIZATION:
    __all__.extend([
        "RobotVisualizer",
        "plot_robot_2d",
        "plot_robot_3d",
        "animate_robot_motion",
        "plot_workspace",
        "plot_jacobian_analysis",
    ])
