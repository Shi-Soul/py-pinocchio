#!/usr/bin/env python3
"""
Basic usage example for py-pinocchio

This example demonstrates the fundamental operations of the py-pinocchio library:
1. Creating a simple robot model programmatically
2. Computing forward kinematics
3. Computing Jacobians
4. Computing inverse dynamics

The example uses a simple 2-DOF planar robot for educational clarity.
"""

import numpy as np
import py_pinocchio as pin

def create_simple_2dof_robot():
    """
    Create a simple 2-DOF planar robot for demonstration.
    
    Robot structure:
    - Base link (fixed)
    - Link 1: connected by revolute joint 1
    - Link 2: connected by revolute joint 2
    
    This represents a simple 2-link planar manipulator.
    """
    print("Creating a simple 2-DOF planar robot...")
    
    # Create links
    base_link = pin.Link(
        name="base_link",
        mass=0.0,  # Base has no mass
        center_of_mass=np.zeros(3),
        inertia_tensor=np.eye(3)
    )
    
    link1 = pin.Link(
        name="link1", 
        mass=1.0,
        center_of_mass=np.array([0.5, 0.0, 0.0]),  # COM at middle of link
        inertia_tensor=np.diag([0.1, 0.1, 0.1])
    )
    
    link2 = pin.Link(
        name="link2",
        mass=0.5,
        center_of_mass=np.array([0.3, 0.0, 0.0]),  # COM at middle of link
        inertia_tensor=np.diag([0.05, 0.05, 0.05])
    )
    
    # Create joints
    joint1 = pin.Joint(
        name="joint1",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),  # Rotate about Z-axis
        parent_link="base_link",
        child_link="link1",
        origin_transform=pin.create_identity_transform(),
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    
    # Joint 2 is offset by 1 meter in X direction from joint 1
    joint2_origin = pin.create_transform(
        rotation=np.eye(3),
        translation=np.array([1.0, 0.0, 0.0])
    )
    
    joint2 = pin.Joint(
        name="joint2",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),  # Rotate about Z-axis
        parent_link="link1", 
        child_link="link2",
        origin_transform=joint2_origin,
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    
    # Create robot model
    robot = pin.create_robot_model(
        name="simple_2dof_robot",
        links=[base_link, link1, link2],
        joints=[joint1, joint2],
        root_link="base_link"
    )
    
    print(f"Created robot with {robot.num_links} links and {robot.num_dof} DOF")
    return robot


def demonstrate_forward_kinematics(robot):
    """Demonstrate forward kinematics computation."""
    print("\n=== Forward Kinematics Demo ===")
    
    # Define joint positions
    joint_positions = np.array([np.pi/4, np.pi/6])  # 45° and 30°
    print(f"Joint positions: {joint_positions} rad")
    print(f"Joint positions: {np.degrees(joint_positions)} deg")
    
    # Compute forward kinematics
    kinematic_state = pin.compute_forward_kinematics(robot, joint_positions)
    
    # Get end-effector position
    end_effector_pos = pin.get_link_position(robot, joint_positions, "link2")
    print(f"End-effector position: {end_effector_pos}")
    
    # Get end-effector orientation
    end_effector_rot = pin.get_link_orientation(robot, joint_positions, "link2")
    print(f"End-effector orientation (rotation matrix):")
    print(end_effector_rot)
    
    # Show all link positions
    print("\nAll link positions:")
    for link_name, transform in kinematic_state.link_transforms.items():
        print(f"  {link_name}: {transform.translation}")


def demonstrate_jacobian_computation(robot):
    """Demonstrate Jacobian computation."""
    print("\n=== Jacobian Computation Demo ===")
    
    joint_positions = np.array([np.pi/4, np.pi/6])
    
    # Compute geometric Jacobian
    jacobian = pin.compute_geometric_jacobian(robot, joint_positions, "link2")
    print(f"Geometric Jacobian shape: {jacobian.shape}")
    print("Geometric Jacobian:")
    print(jacobian)
    
    # Analyze Jacobian properties
    jacobian_computer = pin.JacobianComputer(robot)
    analysis = jacobian_computer.analyze_singularities(joint_positions, "link2")
    
    print(f"\nJacobian Analysis:")
    print(f"  Singular values: {analysis['singular_values']}")
    print(f"  Manipulability: {analysis['manipulability']:.6f}")
    print(f"  Condition number: {analysis['condition_number']:.2f}")
    print(f"  Is singular: {analysis['is_singular']}")


def demonstrate_inverse_dynamics(robot):
    """Demonstrate inverse dynamics computation."""
    print("\n=== Inverse Dynamics Demo ===")
    
    # Define motion
    joint_positions = np.array([np.pi/4, np.pi/6])
    joint_velocities = np.array([0.1, -0.2])  # rad/s
    joint_accelerations = np.array([0.5, 0.3])  # rad/s²
    
    print(f"Joint positions: {joint_positions} rad")
    print(f"Joint velocities: {joint_velocities} rad/s")
    print(f"Joint accelerations: {joint_accelerations} rad/s²")
    
    # Compute required torques
    torques = pin.compute_inverse_dynamics(
        robot, joint_positions, joint_velocities, joint_accelerations
    )
    print(f"Required joint torques: {torques} N⋅m")
    
    # Compute mass matrix
    mass_matrix = pin.compute_mass_matrix(robot, joint_positions)
    print(f"\nMass matrix:")
    print(mass_matrix)
    
    # Compute gravity vector
    gravity_vector = pin.compute_gravity_vector(robot, joint_positions)
    print(f"\nGravity vector: {gravity_vector} N⋅m")


def demonstrate_workspace_analysis(robot):
    """Demonstrate workspace analysis."""
    print("\n=== Workspace Analysis Demo ===")
    
    # Sample workspace
    n_samples = 20
    positions = []
    
    for i in range(n_samples):
        for j in range(n_samples):
            q1 = -np.pi + 2*np.pi * i / (n_samples - 1)
            q2 = -np.pi + 2*np.pi * j / (n_samples - 1)
            
            joint_pos = np.array([q1, q2])
            if robot.is_valid_joint_configuration(joint_pos):
                ee_pos = pin.get_link_position(robot, joint_pos, "link2")
                positions.append(ee_pos[:2])  # Only X-Y for 2D robot
    
    positions = np.array(positions)
    print(f"Sampled {len(positions)} workspace points")
    print(f"Workspace bounds:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")


def main():
    """Main demonstration function."""
    print("py-pinocchio Basic Usage Example")
    print("=" * 40)
    
    # Create robot
    robot = create_simple_2dof_robot()
    
    # Demonstrate various computations
    demonstrate_forward_kinematics(robot)
    demonstrate_jacobian_computation(robot)
    demonstrate_inverse_dynamics(robot)
    demonstrate_workspace_analysis(robot)
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("- Try loading a URDF file with parse_urdf_file()")
    print("- Experiment with different joint configurations")
    print("- Visualize the robot workspace")
    print("- Implement simple control algorithms")


if __name__ == "__main__":
    main()
