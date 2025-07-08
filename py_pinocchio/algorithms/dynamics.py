"""
Rigid body dynamics algorithms using functional programming style

This module implements the core dynamics algorithms for robot manipulators:
1. Inverse Dynamics: Compute joint torques from motion (Recursive Newton-Euler)
2. Forward Dynamics: Compute joint accelerations from torques (Articulated Body Algorithm)
3. Mass Matrix: Compute joint-space inertia matrix
4. Coriolis Matrix: Compute Coriolis and centrifugal effects
5. Gravity Vector: Compute gravitational effects

All algorithms are implemented as pure functions for educational clarity.
The focus is on understanding the physics rather than computational efficiency.
"""

import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass

from ..model import RobotModel, Joint, JointType
from ..math.spatial import (
    SpatialVector, SpatialMatrix, SpatialInertia,
    create_spatial_vector, create_spatial_motion, create_spatial_force,
    spatial_cross_product, spatial_cross_product_dual,
    multiply_spatial_matrix_vector, create_spatial_transformation_matrix
)
from ..math.transform import Transform, invert_transform
from .kinematics import compute_forward_kinematics


class DynamicsState(NamedTuple):
    """
    Immutable dynamics state containing all computed quantities.
    
    This represents the complete dynamic state of the robot including
    spatial velocities, accelerations, and forces for all links.
    """
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    joint_torques: np.ndarray
    
    # Spatial quantities for each link
    link_spatial_velocities: Dict[str, SpatialVector]
    link_spatial_accelerations: Dict[str, SpatialVector]
    link_spatial_forces: Dict[str, SpatialVector]


def compute_inverse_dynamics(robot: RobotModel, 
                           joint_positions: np.ndarray,
                           joint_velocities: np.ndarray, 
                           joint_accelerations: np.ndarray,
                           gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
    """
    Compute inverse dynamics using Recursive Newton-Euler Algorithm.
    
    Given joint positions, velocities, and accelerations, compute the joint
    torques required to produce this motion. This is the fundamental equation:
    τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
    
    Algorithm (Recursive Newton-Euler):
    1. Forward pass: Compute link velocities and accelerations
    2. Backward pass: Compute link forces and joint torques
    
    Args:
        robot: Robot model
        joint_positions: Joint positions (n_dof,)
        joint_velocities: Joint velocities (n_dof,)
        joint_accelerations: Joint accelerations (n_dof,)
        gravity: Gravity vector in world frame (default: [0,0,-9.81])
        
    Returns:
        Joint torques (n_dof,)
        
    Raises:
        ValueError: If input dimensions don't match robot DOF
    """
    n_dof = robot.num_dof
    
    # Validate inputs
    if len(joint_positions) != n_dof:
        raise ValueError(f"Expected {n_dof} joint positions, got {len(joint_positions)}")
    if len(joint_velocities) != n_dof:
        raise ValueError(f"Expected {n_dof} joint velocities, got {len(joint_velocities)}")
    if len(joint_accelerations) != n_dof:
        raise ValueError(f"Expected {n_dof} joint accelerations, got {len(joint_accelerations)}")
    
    # Compute forward kinematics
    kinematic_state = compute_forward_kinematics(robot, joint_positions)
    
    # Create joint state mappings
    actuated_joints = robot.actuated_joint_names
    joint_vel_map = {name: joint_velocities[i] for i, name in enumerate(actuated_joints)}
    joint_acc_map = {name: joint_accelerations[i] for i, name in enumerate(actuated_joints)}
    
    # Initialize spatial quantities
    link_velocities = {}
    link_accelerations = {}
    link_forces = {}
    
    # Forward pass: compute velocities and accelerations
    _forward_dynamics_pass(
        robot=robot,
        kinematic_state=kinematic_state,
        joint_vel_map=joint_vel_map,
        joint_acc_map=joint_acc_map,
        gravity=gravity,
        link_velocities=link_velocities,
        link_accelerations=link_accelerations
    )
    
    # Backward pass: compute forces and torques
    joint_torques = np.zeros(n_dof)
    _backward_dynamics_pass(
        robot=robot,
        kinematic_state=kinematic_state,
        link_velocities=link_velocities,
        link_accelerations=link_accelerations,
        link_forces=link_forces,
        joint_torques=joint_torques,
        actuated_joints=actuated_joints
    )
    
    return joint_torques


def _forward_dynamics_pass(robot: RobotModel,
                          kinematic_state,
                          joint_vel_map: Dict[str, float],
                          joint_acc_map: Dict[str, float],
                          gravity: np.ndarray,
                          link_velocities: Dict[str, SpatialVector],
                          link_accelerations: Dict[str, SpatialVector]) -> None:
    """
    Forward pass of Newton-Euler algorithm: compute link velocities and accelerations.
    
    This pass propagates motion from the base to the end-effectors.
    """
    # Root link starts with zero velocity (assuming fixed base)
    if robot.root_link:
        link_velocities[robot.root_link] = create_spatial_motion(
            angular_velocity=np.zeros(3),
            linear_velocity=np.zeros(3)
        )
        
        # Root acceleration includes gravity effect
        gravity_acceleration = create_spatial_motion(
            angular_velocity=np.zeros(3),
            linear_velocity=-gravity  # Negative because we want upward acceleration
        )
        link_accelerations[robot.root_link] = gravity_acceleration
    
    # Traverse tree and compute velocities/accelerations
    _traverse_forward_dynamics(
        robot=robot,
        current_link=robot.root_link,
        kinematic_state=kinematic_state,
        joint_vel_map=joint_vel_map,
        joint_acc_map=joint_acc_map,
        link_velocities=link_velocities,
        link_accelerations=link_accelerations
    )


def _traverse_forward_dynamics(robot: RobotModel,
                              current_link: str,
                              kinematic_state,
                              joint_vel_map: Dict[str, float],
                              joint_acc_map: Dict[str, float],
                              link_velocities: Dict[str, SpatialVector],
                              link_accelerations: Dict[str, SpatialVector]) -> None:
    """Recursively traverse tree for forward dynamics pass."""
    if not current_link:
        return
    
    current_velocity = link_velocities.get(current_link, create_spatial_motion(np.zeros(3), np.zeros(3)))
    current_acceleration = link_accelerations.get(current_link, create_spatial_motion(np.zeros(3), np.zeros(3)))
    
    # Process all child links
    for child_link in robot.get_child_links(current_link):
        joint = robot.get_joint_between_links(current_link, child_link)
        if not joint:
            continue
        
        # Compute child velocity and acceleration
        child_velocity, child_acceleration = _compute_child_motion(
            joint=joint,
            parent_velocity=current_velocity,
            parent_acceleration=current_acceleration,
            joint_vel_map=joint_vel_map,
            joint_acc_map=joint_acc_map,
            kinematic_state=kinematic_state
        )
        
        link_velocities[child_link] = child_velocity
        link_accelerations[child_link] = child_acceleration
        
        # Recursively process children
        _traverse_forward_dynamics(
            robot=robot,
            current_link=child_link,
            kinematic_state=kinematic_state,
            joint_vel_map=joint_vel_map,
            joint_acc_map=joint_acc_map,
            link_velocities=link_velocities,
            link_accelerations=link_accelerations
        )


def _compute_child_motion(joint: Joint,
                         parent_velocity: SpatialVector,
                         parent_acceleration: SpatialVector,
                         joint_vel_map: Dict[str, float],
                         joint_acc_map: Dict[str, float],
                         kinematic_state) -> Tuple[SpatialVector, SpatialVector]:
    """
    Compute child link motion from parent motion and joint motion.
    
    This implements the kinematic relationships for spatial velocities
    and accelerations across joints.
    """
    if joint.joint_type == JointType.FIXED:
        # Fixed joint: child has same motion as parent
        return parent_velocity, parent_acceleration
    
    elif joint.joint_type == JointType.REVOLUTE:
        # Revolute joint motion
        joint_vel = joint_vel_map.get(joint.name, 0.0)
        joint_acc = joint_acc_map.get(joint.name, 0.0)
        
        # Joint motion in spatial coordinates
        joint_motion_vel = create_spatial_motion(
            angular_velocity=joint_vel * joint.axis,
            linear_velocity=np.zeros(3)
        )
        
        joint_motion_acc = create_spatial_motion(
            angular_velocity=joint_acc * joint.axis,
            linear_velocity=np.zeros(3)
        )
        
        # Child velocity: parent + joint
        from ..math.spatial import add_spatial_vectors
        child_velocity = add_spatial_vectors(parent_velocity, joint_motion_vel)
        
        # Child acceleration: parent + joint + velocity coupling
        velocity_coupling = spatial_cross_product(parent_velocity, joint_motion_vel)
        child_acceleration = add_spatial_vectors(
            add_spatial_vectors(parent_acceleration, joint_motion_acc),
            velocity_coupling
        )
        
        return child_velocity, child_acceleration
    
    elif joint.joint_type == JointType.PRISMATIC:
        # Prismatic joint motion
        joint_vel = joint_vel_map.get(joint.name, 0.0)
        joint_acc = joint_acc_map.get(joint.name, 0.0)
        
        # Joint motion in spatial coordinates
        joint_motion_vel = create_spatial_motion(
            angular_velocity=np.zeros(3),
            linear_velocity=joint_vel * joint.axis
        )
        
        joint_motion_acc = create_spatial_motion(
            angular_velocity=np.zeros(3),
            linear_velocity=joint_acc * joint.axis
        )
        
        # Child velocity: parent + joint
        from ..math.spatial import add_spatial_vectors
        child_velocity = add_spatial_vectors(parent_velocity, joint_motion_vel)
        
        # Child acceleration: parent + joint + velocity coupling
        velocity_coupling = spatial_cross_product(parent_velocity, joint_motion_vel)
        child_acceleration = add_spatial_vectors(
            add_spatial_vectors(parent_acceleration, joint_motion_acc),
            velocity_coupling
        )
        
        return child_velocity, child_acceleration
    
    else:
        # Unsupported joint type
        return parent_velocity, parent_acceleration


def _backward_dynamics_pass(robot: RobotModel,
                           kinematic_state,
                           link_velocities: Dict[str, SpatialVector],
                           link_accelerations: Dict[str, SpatialVector],
                           link_forces: Dict[str, SpatialVector],
                           joint_torques: np.ndarray,
                           actuated_joints: List[str]) -> None:
    """
    Backward pass of Newton-Euler algorithm: compute forces and torques.
    
    This pass propagates forces from end-effectors back to the base.
    """
    # Start from leaves and work backwards
    _traverse_backward_dynamics(
        robot=robot,
        current_link=robot.root_link,
        kinematic_state=kinematic_state,
        link_velocities=link_velocities,
        link_accelerations=link_accelerations,
        link_forces=link_forces,
        joint_torques=joint_torques,
        actuated_joints=actuated_joints
    )


def _traverse_backward_dynamics(robot: RobotModel,
                               current_link: str,
                               kinematic_state,
                               link_velocities: Dict[str, SpatialVector],
                               link_accelerations: Dict[str, SpatialVector],
                               link_forces: Dict[str, SpatialVector],
                               joint_torques: np.ndarray,
                               actuated_joints: List[str]) -> None:
    """Recursively traverse tree for backward dynamics pass."""
    if not current_link:
        return
    
    # First, recursively process all children
    for child_link in robot.get_child_links(current_link):
        _traverse_backward_dynamics(
            robot=robot,
            current_link=child_link,
            kinematic_state=kinematic_state,
            link_velocities=link_velocities,
            link_accelerations=link_accelerations,
            link_forces=link_forces,
            joint_torques=joint_torques,
            actuated_joints=actuated_joints
        )
    
    # Compute force for current link
    link = robot.get_link(current_link)
    if not link:
        return
    
    # Inertial force: I * a + v × (I * v)
    link_velocity = link_velocities.get(current_link, create_spatial_motion(np.zeros(3), np.zeros(3)))
    link_acceleration = link_accelerations.get(current_link, create_spatial_motion(np.zeros(3), np.zeros(3)))
    
    # Simplified inertial force computation (assuming link inertia is available)
    inertial_force = _compute_inertial_force(link, link_velocity, link_acceleration)
    
    # Add forces from child links (transmitted through joints)
    total_force = inertial_force
    for child_link in robot.get_child_links(current_link):
        child_force = link_forces.get(child_link, create_spatial_force(np.zeros(3), np.zeros(3)))
        from ..math.spatial import add_spatial_vectors
        total_force = add_spatial_vectors(total_force, child_force)
    
    link_forces[current_link] = total_force
    
    # Compute joint torque if this link is connected by an actuated joint
    parent_link = robot.get_parent_link(current_link)
    if parent_link:
        joint = robot.get_joint_between_links(parent_link, current_link)
        if joint and joint.name in actuated_joints:
            joint_index = actuated_joints.index(joint.name)
            joint_torques[joint_index] = _compute_joint_torque(joint, total_force)


def _compute_inertial_force(link, velocity: SpatialVector, acceleration: SpatialVector) -> SpatialVector:
    """
    Compute inertial force for a link: I*a + v×(I*v)

    This computes the spatial force required to produce the given acceleration,
    including both translational and rotational inertial effects.
    """
    if not link.has_inertia:
        return create_spatial_force(np.zeros(3), np.zeros(3))

    # Get spatial inertia matrix for the link
    spatial_inertia = link.spatial_inertia

    # Compute I*a (inertial force due to acceleration)
    inertial_force_accel = multiply_spatial_matrix_vector(spatial_inertia, acceleration)

    # Compute v×(I*v) (gyroscopic/Coriolis force due to velocity)
    I_times_v = multiply_spatial_matrix_vector(spatial_inertia, velocity)
    gyroscopic_force = spatial_cross_product_dual(velocity, I_times_v)

    # Total inertial force: I*a + v×(I*v)
    from ..math.spatial import add_spatial_vectors
    total_inertial_force = add_spatial_vectors(inertial_force_accel, gyroscopic_force)

    return total_inertial_force


def _compute_joint_torque(joint: Joint, force: SpatialVector) -> float:
    """
    Compute joint torque from spatial force.
    
    For revolute joint: τ = axis^T * moment
    For prismatic joint: τ = axis^T * force
    """
    if joint.joint_type == JointType.REVOLUTE:
        return np.dot(joint.axis, force.angular)
    elif joint.joint_type == JointType.PRISMATIC:
        return np.dot(joint.axis, force.linear)
    else:
        return 0.0


def compute_mass_matrix(robot: RobotModel, joint_positions: np.ndarray) -> np.ndarray:
    """
    Compute joint-space mass matrix M(q).
    
    The mass matrix relates joint accelerations to joint torques:
    τ = M(q)q̈ + ...
    
    This is computed by setting velocities to zero and accelerations
    to unit vectors, then computing the resulting torques.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        
    Returns:
        n_dof x n_dof mass matrix
    """
    n_dof = robot.num_dof
    mass_matrix = np.zeros((n_dof, n_dof))
    
    # Zero velocities
    zero_velocities = np.zeros(n_dof)
    
    # Compute each column by setting one acceleration to 1, others to 0
    for i in range(n_dof):
        unit_acceleration = np.zeros(n_dof)
        unit_acceleration[i] = 1.0
        
        # Compute torques for this unit acceleration
        torques = compute_inverse_dynamics(
            robot, joint_positions, zero_velocities, unit_acceleration,
            gravity=np.zeros(3)  # No gravity for mass matrix
        )
        
        mass_matrix[:, i] = torques
    
    return mass_matrix


def compute_gravity_vector(robot: RobotModel, joint_positions: np.ndarray,
                          gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
    """
    Compute gravity vector g(q).
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        gravity: Gravity vector
        
    Returns:
        n_dof gravity vector
    """
    n_dof = robot.num_dof
    zero_velocities = np.zeros(n_dof)
    zero_accelerations = np.zeros(n_dof)
    
    return compute_inverse_dynamics(
        robot, joint_positions, zero_velocities, zero_accelerations, gravity
    )


@dataclass
class InverseDynamics:
    """Object-oriented interface for inverse dynamics."""
    robot: RobotModel
    
    def compute(self, joint_positions: np.ndarray, joint_velocities: np.ndarray,
                joint_accelerations: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics."""
        return compute_inverse_dynamics(
            self.robot, joint_positions, joint_velocities, joint_accelerations
        )


@dataclass
class ForwardDynamics:
    """Object-oriented interface for forward dynamics."""
    robot: RobotModel

    def compute(self, joint_positions: np.ndarray, joint_velocities: np.ndarray,
                joint_torques: np.ndarray, gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
        """Compute forward dynamics."""
        return compute_forward_dynamics(
            self.robot, joint_positions, joint_velocities, joint_torques, gravity
        )

    def compute_mass_matrix(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute mass matrix at given configuration."""
        return compute_mass_matrix(self.robot, joint_positions)

    def compute_coriolis_forces(self, joint_positions: np.ndarray,
                               joint_velocities: np.ndarray) -> np.ndarray:
        """Compute Coriolis and centrifugal forces."""
        return compute_coriolis_forces(self.robot, joint_positions, joint_velocities)

    def compute_gravity_forces(self, joint_positions: np.ndarray,
                              gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
        """Compute gravity forces."""
        return compute_gravity_vector(self.robot, joint_positions, gravity)


def compute_forward_dynamics(robot: RobotModel, joint_positions: np.ndarray,
                           joint_velocities: np.ndarray, joint_torques: np.ndarray,
                           gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
    """
    Compute forward dynamics using the equation of motion.

    Solves: q̈ = M(q)^(-1) * (τ - C(q,q̇)q̇ - g(q))

    This is a simplified implementation that uses the mass matrix approach
    rather than the more efficient Articulated Body Algorithm (ABA).

    Args:
        robot: Robot model
        joint_positions: Joint positions (n_dof,)
        joint_velocities: Joint velocities (n_dof,)
        joint_torques: Applied joint torques (n_dof,)
        gravity: Gravity vector in world frame

    Returns:
        Joint accelerations (n_dof,)

    Raises:
        ValueError: If input dimensions don't match robot DOF
    """
    n_dof = robot.num_dof

    # Validate inputs
    if len(joint_positions) != n_dof:
        raise ValueError(f"Expected {n_dof} joint positions, got {len(joint_positions)}")
    if len(joint_velocities) != n_dof:
        raise ValueError(f"Expected {n_dof} joint velocities, got {len(joint_velocities)}")
    if len(joint_torques) != n_dof:
        raise ValueError(f"Expected {n_dof} joint torques, got {len(joint_torques)}")

    # Compute mass matrix M(q)
    mass_matrix = compute_mass_matrix(robot, joint_positions)

    # Compute Coriolis and centrifugal forces C(q,q̇)q̇
    coriolis_forces = compute_coriolis_forces(robot, joint_positions, joint_velocities)

    # Compute gravity forces g(q)
    gravity_forces = compute_gravity_vector(robot, joint_positions, gravity)

    # Solve: M(q)q̈ = τ - C(q,q̇)q̇ - g(q)
    # Therefore: q̈ = M(q)^(-1) * (τ - C(q,q̇)q̇ - g(q))

    # Right-hand side of equation
    rhs = joint_torques - coriolis_forces - gravity_forces

    # Solve for accelerations
    try:
        # Use pseudo-inverse for numerical stability
        mass_matrix_inv = np.linalg.pinv(mass_matrix, rcond=1e-6)
        joint_accelerations = mass_matrix_inv @ rhs
    except np.linalg.LinAlgError:
        # Fallback to least squares if matrix is singular
        joint_accelerations = np.linalg.lstsq(mass_matrix, rhs, rcond=None)[0]

    return joint_accelerations


def compute_coriolis_forces(robot: RobotModel, joint_positions: np.ndarray,
                          joint_velocities: np.ndarray) -> np.ndarray:
    """
    Compute Coriolis and centrifugal forces C(q,q̇)q̇.

    This uses numerical differentiation of the mass matrix to compute
    the Coriolis effects. The Coriolis matrix C(q,q̇) satisfies:
    C(q,q̇)q̇ = Ṁ(q)q̇ - (1/2) * ∂/∂q(q̇ᵀM(q)q̇)

    Args:
        robot: Robot model
        joint_positions: Joint positions (n_dof,)
        joint_velocities: Joint velocities (n_dof,)

    Returns:
        Coriolis and centrifugal forces (n_dof,)
    """
    n_dof = robot.num_dof

    if len(joint_positions) != n_dof or len(joint_velocities) != n_dof:
        raise ValueError("Position and velocity vectors must match robot DOF")

    # For educational simplicity, use numerical differentiation
    # In practice, this would use more efficient symbolic computation

    dt = 1e-8  # Very small time step for numerical differentiation

    # Current mass matrix
    M_current = compute_mass_matrix(robot, joint_positions)

    # Mass matrix at slightly perturbed positions
    q_perturbed = joint_positions + dt * joint_velocities
    M_perturbed = compute_mass_matrix(robot, q_perturbed)

    # Time derivative of mass matrix: Ṁ ≈ (M_perturbed - M_current) / dt
    M_dot = (M_perturbed - M_current) / dt

    # Coriolis forces: C(q,q̇)q̇ ≈ Ṁ(q)q̇ - (1/2) * ∂/∂q(q̇ᵀM(q)q̇)
    coriolis_term1 = M_dot @ joint_velocities

    # Second term: (1/2) * ∂/∂q(q̇ᵀM(q)q̇)
    # This is computed using numerical gradients
    coriolis_term2 = np.zeros(n_dof)

    for i in range(n_dof):
        # Perturb each joint position slightly (with bounds checking)
        q_plus = joint_positions.copy()
        q_minus = joint_positions.copy()

        # Check joint limits to avoid exceeding them
        joint_name = robot.actuated_joint_names[i]
        joint = robot.get_joint(joint_name)
        if joint:
            q_min = joint.position_limit_lower
            q_max = joint.position_limit_upper

            # Adjust perturbation to stay within limits
            dt_plus = min(dt, q_max - joint_positions[i]) if q_max < np.inf else dt
            dt_minus = min(dt, joint_positions[i] - q_min) if q_min > -np.inf else dt

            q_plus[i] += dt_plus
            q_minus[i] -= dt_minus
        else:
            q_plus[i] += dt
            q_minus[i] -= dt

        # Compute kinetic energy at perturbed positions
        M_plus = compute_mass_matrix(robot, q_plus)
        M_minus = compute_mass_matrix(robot, q_minus)

        kinetic_plus = 0.5 * joint_velocities.T @ M_plus @ joint_velocities
        kinetic_minus = 0.5 * joint_velocities.T @ M_minus @ joint_velocities

        # Numerical gradient
        coriolis_term2[i] = (kinetic_plus - kinetic_minus) / (2 * dt)

    # Combine terms
    coriolis_forces = coriolis_term1 - coriolis_term2

    return coriolis_forces


def compute_coriolis_matrix(robot: RobotModel, joint_positions: np.ndarray,
                           joint_velocities: np.ndarray) -> np.ndarray:
    """
    Compute Coriolis matrix C(q,q̇) such that C(q,q̇)q̇ gives Coriolis forces.

    This is computed by numerical differentiation for educational purposes.
    In practice, more efficient symbolic methods would be used.

    Args:
        robot: Robot model
        joint_positions: Joint positions
        joint_velocities: Joint velocities

    Returns:
        n_dof x n_dof Coriolis matrix
    """
    n_dof = robot.num_dof
    coriolis_matrix = np.zeros((n_dof, n_dof))

    # Compute each column by setting one velocity to 1, others to 0
    for j in range(n_dof):
        unit_velocity = np.zeros(n_dof)
        unit_velocity[j] = 1.0

        # Compute Coriolis forces for this unit velocity
        coriolis_forces = compute_coriolis_forces(robot, joint_positions, unit_velocity)
        coriolis_matrix[:, j] = coriolis_forces

    return coriolis_matrix
