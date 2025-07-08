"""
Jacobian computation algorithms using functional programming style

This module implements Jacobian matrix computations for robot manipulators.
The Jacobian relates joint velocities to end-effector velocities and is
fundamental for velocity control, singularity analysis, and dynamics.

Two types of Jacobians are implemented:
1. Geometric Jacobian: Relates joint velocities to spatial velocity (twist)
2. Analytical Jacobian: Relates joint velocities to task-space velocities

All algorithms are implemented as pure functions for educational clarity.
"""

import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

from ..model import RobotModel, Joint, JointType
from ..math.transform import Transform, transform_vector
from ..math.spatial import SpatialVector, create_spatial_vector
from ..math.utils import create_skew_symmetric_matrix
from .kinematics import compute_forward_kinematics, get_link_transform


class JacobianResult(NamedTuple):
    """
    Immutable result of Jacobian computation.
    
    Contains both geometric and analytical Jacobians along with
    metadata about the computation.
    """
    geometric_jacobian: np.ndarray      # 6 x n_dof matrix
    analytical_jacobian: np.ndarray     # 6 x n_dof matrix (same as geometric for now)
    joint_positions: np.ndarray         # Joint positions used
    end_effector_link: str              # Target link name
    reference_frame: str                # Frame in which Jacobian is expressed


def compute_geometric_jacobian(robot: RobotModel, joint_positions: np.ndarray, 
                             end_effector_link: str, 
                             reference_frame: str = "world") -> np.ndarray:
    """
    Compute geometric Jacobian matrix.
    
    The geometric Jacobian J_g relates joint velocities to spatial velocity:
    v_spatial = J_g * q_dot
    
    where v_spatial = [angular_velocity; linear_velocity] is the 6D spatial velocity.
    
    Algorithm:
    1. Compute forward kinematics to get all link transforms
    2. For each actuated joint, compute its contribution to end-effector velocity
    3. Revolute joint: contributes angular velocity about joint axis
    4. Prismatic joint: contributes linear velocity along joint axis
    
    Args:
        robot: Robot model
        joint_positions: Current joint positions
        end_effector_link: Name of end-effector link
        reference_frame: Frame to express Jacobian in ("world" or link name)
        
    Returns:
        6 x n_dof Jacobian matrix
        
    Raises:
        ValueError: If end_effector_link doesn't exist
    """
    if not robot.get_link(end_effector_link):
        raise ValueError(f"End-effector link '{end_effector_link}' not found")
    
    # Compute forward kinematics
    kinematic_state = compute_forward_kinematics(robot, joint_positions)
    
    # Get end-effector transform
    ee_transform = kinematic_state.link_transforms[end_effector_link]
    ee_position = ee_transform.translation
    
    # Initialize Jacobian matrix
    n_dof = robot.num_dof
    jacobian = np.zeros((6, n_dof))
    
    # Process each actuated joint
    actuated_joints = robot.actuated_joint_names
    
    for i, joint_name in enumerate(actuated_joints):
        joint = robot.get_joint(joint_name)
        if not joint:
            continue
        
        # Check if this joint affects the end-effector
        if _joint_affects_link(robot, joint_name, end_effector_link):
            # Compute joint's contribution to Jacobian
            jacobian_column = _compute_joint_jacobian_column(
                joint=joint,
                joint_transform=kinematic_state.joint_transforms[joint_name],
                ee_position=ee_position,
                reference_frame=reference_frame
            )
            jacobian[:, i] = jacobian_column
    
    return jacobian


def _joint_affects_link(robot: RobotModel, joint_name: str, target_link: str) -> bool:
    """
    Check if a joint affects the position/orientation of a target link.
    
    A joint affects a link if the link is in the subtree rooted at the joint's child link.
    """
    joint = robot.get_joint(joint_name)
    if not joint:
        return False
    
    # Perform breadth-first search from joint's child link
    queue = [joint.child_link]
    visited = set()
    
    while queue:
        current_link = queue.pop(0)
        if current_link == target_link:
            return True
        
        if current_link in visited:
            continue
        visited.add(current_link)
        
        # Add all children to queue
        children = robot.get_child_links(current_link)
        queue.extend(children)
    
    return False


def _compute_joint_jacobian_column(joint: Joint, joint_transform: Transform,
                                 ee_position: np.ndarray, reference_frame: str) -> np.ndarray:
    """
    Compute a single column of the Jacobian matrix for one joint.
    
    This represents how the end-effector spatial velocity changes with
    respect to this joint's velocity.
    
    Args:
        joint: Joint to compute column for
        joint_transform: Transform of joint in world frame
        ee_position: End-effector position in world frame
        reference_frame: Reference frame for Jacobian
        
    Returns:
        6D column vector [angular_contribution; linear_contribution]
    """
    if joint.joint_type == JointType.REVOLUTE:
        return _compute_revolute_jacobian_column(joint, joint_transform, ee_position)
    elif joint.joint_type == JointType.PRISMATIC:
        return _compute_prismatic_jacobian_column(joint, joint_transform)
    else:
        # Fixed or unsupported joint types contribute zero
        return np.zeros(6)


def _compute_revolute_jacobian_column(joint: Joint, joint_transform: Transform,
                                    ee_position: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian column for revolute joint.
    
    For a revolute joint:
    - Angular velocity contribution: joint axis in world frame
    - Linear velocity contribution: axis × (ee_pos - joint_pos)
    
    This follows from the fact that revolute motion creates both
    angular velocity and linear velocity due to the lever arm.
    """
    # Joint axis in world frame
    joint_axis_world = transform_vector(joint_transform, joint.axis)
    
    # Joint position in world frame
    joint_position = joint_transform.translation
    
    # Vector from joint to end-effector
    joint_to_ee = ee_position - joint_position
    
    # Angular contribution: joint axis
    angular_contribution = joint_axis_world
    
    # Linear contribution: axis × (ee_pos - joint_pos)
    linear_contribution = np.cross(joint_axis_world, joint_to_ee)
    
    return np.concatenate([angular_contribution, linear_contribution])


def _compute_prismatic_jacobian_column(joint: Joint, joint_transform: Transform) -> np.ndarray:
    """
    Compute Jacobian column for prismatic joint.
    
    For a prismatic joint:
    - Angular velocity contribution: zero (no rotation)
    - Linear velocity contribution: joint axis in world frame
    """
    # Joint axis in world frame
    joint_axis_world = transform_vector(joint_transform, joint.axis)
    
    # Angular contribution: zero
    angular_contribution = np.zeros(3)
    
    # Linear contribution: joint axis
    linear_contribution = joint_axis_world
    
    return np.concatenate([angular_contribution, linear_contribution])


def compute_analytical_jacobian(robot: RobotModel, joint_positions: np.ndarray,
                               end_effector_link: str) -> np.ndarray:
    """
    Compute analytical Jacobian matrix.
    
    The analytical Jacobian relates joint velocities to task-space velocities
    in a specific parameterization (e.g., position + Euler angles).
    
    For educational simplicity, this currently returns the same as geometric Jacobian.
    In practice, analytical Jacobians depend on the chosen orientation representation.
    
    Args:
        robot: Robot model
        joint_positions: Current joint positions
        end_effector_link: Name of end-effector link
        
    Returns:
        6 x n_dof analytical Jacobian matrix
    """
    # For now, return geometric Jacobian
    # TODO: Implement proper analytical Jacobian with orientation parameterization
    return compute_geometric_jacobian(robot, joint_positions, end_effector_link)


def compute_jacobian_time_derivative(robot: RobotModel, joint_positions: np.ndarray,
                                   joint_velocities: np.ndarray, 
                                   end_effector_link: str) -> np.ndarray:
    """
    Compute time derivative of Jacobian matrix.
    
    The Jacobian time derivative J_dot is needed for computing end-effector
    accelerations and for certain control algorithms.
    
    This is computed using numerical differentiation for educational simplicity.
    
    Args:
        robot: Robot model
        joint_positions: Current joint positions
        joint_velocities: Current joint velocities
        end_effector_link: Name of end-effector link
        
    Returns:
        6 x n_dof Jacobian time derivative matrix
    """
    # Numerical differentiation with small time step
    dt = 1e-6
    
    # Current Jacobian
    J_current = compute_geometric_jacobian(robot, joint_positions, end_effector_link)
    
    # Jacobian at next time step
    q_next = joint_positions + dt * joint_velocities
    J_next = compute_geometric_jacobian(robot, q_next, end_effector_link)
    
    # Numerical derivative
    J_dot = (J_next - J_current) / dt
    
    return J_dot


def compute_jacobian_pseudoinverse(jacobian: np.ndarray, damping: float = 1e-6) -> np.ndarray:
    """
    Compute damped pseudoinverse of Jacobian matrix.
    
    The pseudoinverse is used for inverse kinematics and redundancy resolution.
    Damping helps with numerical stability near singularities.
    
    Args:
        jacobian: Jacobian matrix to invert
        damping: Damping factor for numerical stability
        
    Returns:
        Pseudoinverse matrix
    """
    m, n = jacobian.shape
    
    if m >= n:
        # Overdetermined case: J^+ = (J^T J + λI)^(-1) J^T
        return np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(n), jacobian.T)
    else:
        # Underdetermined case: J^+ = J^T (J J^T + λI)^(-1)
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(m), np.eye(m))


def analyze_jacobian_singularities(jacobian: np.ndarray, threshold: float = 1e-3) -> Dict:
    """
    Analyze Jacobian for singularities and manipulability.
    
    Args:
        jacobian: Jacobian matrix to analyze
        threshold: Threshold for singularity detection
        
    Returns:
        Dictionary with singularity analysis results
    """
    # Compute singular values
    U, s, Vt = np.linalg.svd(jacobian)
    
    # Manipulability measure
    manipulability = np.prod(s)
    
    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    
    # Check for singularities
    is_singular = np.any(s < threshold)
    
    return {
        'singular_values': s,
        'manipulability': manipulability,
        'condition_number': condition_number,
        'is_singular': is_singular,
        'rank': np.sum(s > threshold)
    }


@dataclass
class JacobianComputer:
    """
    Object-oriented interface for Jacobian computations.
    
    This class provides a stateful interface while using pure functions internally.
    """
    robot: RobotModel
    
    def compute_geometric(self, joint_positions: np.ndarray, 
                         end_effector_link: str) -> np.ndarray:
        """Compute geometric Jacobian."""
        return compute_geometric_jacobian(self.robot, joint_positions, end_effector_link)
    
    def compute_analytical(self, joint_positions: np.ndarray,
                          end_effector_link: str) -> np.ndarray:
        """Compute analytical Jacobian."""
        return compute_analytical_jacobian(self.robot, joint_positions, end_effector_link)
    
    def compute_time_derivative(self, joint_positions: np.ndarray,
                               joint_velocities: np.ndarray,
                               end_effector_link: str) -> np.ndarray:
        """Compute Jacobian time derivative."""
        return compute_jacobian_time_derivative(
            self.robot, joint_positions, joint_velocities, end_effector_link
        )
    
    def analyze_singularities(self, joint_positions: np.ndarray,
                             end_effector_link: str) -> Dict:
        """Analyze Jacobian for singularities."""
        jacobian = self.compute_geometric(joint_positions, end_effector_link)
        return analyze_jacobian_singularities(jacobian)
