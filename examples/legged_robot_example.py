#!/usr/bin/env python3
"""
Legged robot example for py-pinocchio

This example demonstrates modeling and analysis of a simple legged robot.
We'll create a 2D biped with hip and knee joints for each leg.
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin


def create_simple_biped():
    """
    Create a simple 2D biped robot.
    
    Structure:
    - Torso (base)
    - Left leg: hip joint -> thigh -> knee joint -> shin -> foot
    - Right leg: hip joint -> thigh -> knee joint -> shin -> foot
    
    Total: 4 DOF (2 hips + 2 knees)
    """
    print("Creating simple biped robot...")
    
    # Physical parameters
    torso_mass = 10.0
    thigh_mass = 2.0
    shin_mass = 1.5
    foot_mass = 0.5
    
    thigh_length = 0.4
    shin_length = 0.35
    foot_length = 0.2
    hip_width = 0.3
    
    # Create links
    links = []
    
    # Torso (base)
    torso = pin.Link(
        name="torso",
        mass=torso_mass,
        center_of_mass=np.array([0.0, 0.0, 0.1]),
        inertia_tensor=np.diag([0.5, 0.5, 0.3])
    )
    links.append(torso)
    
    # Left leg links
    left_thigh = pin.Link(
        name="left_thigh",
        mass=thigh_mass,
        center_of_mass=np.array([0.0, 0.0, -thigh_length/2]),
        inertia_tensor=np.diag([0.1, 0.1, 0.02])
    )
    links.append(left_thigh)
    
    left_shin = pin.Link(
        name="left_shin", 
        mass=shin_mass,
        center_of_mass=np.array([0.0, 0.0, -shin_length/2]),
        inertia_tensor=np.diag([0.08, 0.08, 0.01])
    )
    links.append(left_shin)
    
    left_foot = pin.Link(
        name="left_foot",
        mass=foot_mass,
        center_of_mass=np.array([foot_length/2, 0.0, 0.0]),
        inertia_tensor=np.diag([0.01, 0.02, 0.01])
    )
    links.append(left_foot)
    
    # Right leg links (mirror of left)
    right_thigh = pin.Link(
        name="right_thigh",
        mass=thigh_mass,
        center_of_mass=np.array([0.0, 0.0, -thigh_length/2]),
        inertia_tensor=np.diag([0.1, 0.1, 0.02])
    )
    links.append(right_thigh)
    
    right_shin = pin.Link(
        name="right_shin",
        mass=shin_mass,
        center_of_mass=np.array([0.0, 0.0, -shin_length/2]),
        inertia_tensor=np.diag([0.08, 0.08, 0.01])
    )
    links.append(right_shin)
    
    right_foot = pin.Link(
        name="right_foot",
        mass=foot_mass,
        center_of_mass=np.array([foot_length/2, 0.0, 0.0]),
        inertia_tensor=np.diag([0.01, 0.02, 0.01])
    )
    links.append(right_foot)
    
    # Create joints
    joints = []
    
    # Left leg joints
    left_hip = pin.Joint(
        name="left_hip",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),  # Rotation about Y-axis (sagittal plane)
        parent_link="torso",
        child_link="left_thigh",
        origin_transform=pin.create_transform(np.eye(3), np.array([-hip_width/2, 0, 0])),
        position_limit_lower=-np.pi/2,
        position_limit_upper=np.pi/2
    )
    joints.append(left_hip)
    
    left_knee = pin.Joint(
        name="left_knee",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="left_thigh",
        child_link="left_shin",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -thigh_length])),
        position_limit_lower=0.0,  # Knee can only bend forward
        position_limit_upper=np.pi
    )
    joints.append(left_knee)
    
    left_ankle = pin.Joint(
        name="left_ankle",
        joint_type=pin.JointType.FIXED,  # Fixed for simplicity
        parent_link="left_shin",
        child_link="left_foot",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -shin_length]))
    )
    joints.append(left_ankle)
    
    # Right leg joints (mirror of left)
    right_hip = pin.Joint(
        name="right_hip",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="torso",
        child_link="right_thigh",
        origin_transform=pin.create_transform(np.eye(3), np.array([hip_width/2, 0, 0])),
        position_limit_lower=-np.pi/2,
        position_limit_upper=np.pi/2
    )
    joints.append(right_hip)
    
    right_knee = pin.Joint(
        name="right_knee",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="right_thigh",
        child_link="right_shin",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -thigh_length])),
        position_limit_lower=0.0,
        position_limit_upper=np.pi
    )
    joints.append(right_knee)
    
    right_ankle = pin.Joint(
        name="right_ankle",
        joint_type=pin.JointType.FIXED,
        parent_link="right_shin",
        child_link="right_foot",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -shin_length]))
    )
    joints.append(right_ankle)
    
    # Create robot model
    robot = pin.create_robot_model(
        name="simple_biped",
        links=links,
        joints=joints,
        root_link="torso"
    )
    
    print(f"Created biped with {robot.num_links} links and {robot.num_dof} DOF")
    print(f"Actuated joints: {robot.actuated_joint_names}")
    
    return robot


def demonstrate_biped_kinematics(robot):
    """Demonstrate forward kinematics for different walking poses."""
    print("\n=== Biped Kinematics Analysis ===")
    
    # Define walking poses
    poses = {
        "standing": np.array([0.0, 0.0, 0.0, 0.0]),  # All joints at zero
        "left_step": np.array([0.3, 0.5, -0.2, 0.3]),  # Left leg forward
        "right_step": np.array([-0.2, 0.3, 0.3, 0.5]),  # Right leg forward
        "squat": np.array([0.5, 1.0, 0.5, 1.0]),  # Both knees bent
        "lunge": np.array([0.8, 1.2, -0.3, 0.2]),  # Deep lunge
    }
    
    # Analyze each pose
    for pose_name, joint_config in poses.items():
        print(f"\nPose: {pose_name}")
        print(f"  Joint configuration: {joint_config}")
        
        try:
            # Compute foot positions
            left_foot_pos = pin.get_link_position(robot, joint_config, "left_foot")
            right_foot_pos = pin.get_link_position(robot, joint_config, "right_foot")
            
            print(f"  Left foot position: [{left_foot_pos[0]:.3f}, {left_foot_pos[1]:.3f}, {left_foot_pos[2]:.3f}]")
            print(f"  Right foot position: [{right_foot_pos[0]:.3f}, {right_foot_pos[1]:.3f}, {right_foot_pos[2]:.3f}]")
            
            # Check stability (simple criterion: both feet below torso)
            torso_pos = pin.get_link_position(robot, joint_config, "torso")
            stable = (left_foot_pos[2] < torso_pos[2]) and (right_foot_pos[2] < torso_pos[2])
            print(f"  Stable pose: {stable}")
            
            # Compute center of mass
            com = compute_center_of_mass(robot, joint_config)
            print(f"  Center of mass: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
            
        except Exception as e:
            print(f"  Error: {e}")


def compute_center_of_mass(robot, joint_positions):
    """Compute center of mass of the robot."""
    kinematic_state = pin.compute_forward_kinematics(robot, joint_positions)
    
    total_mass = 0.0
    weighted_position = np.zeros(3)
    
    for link in robot.links:
        if link.has_inertia:
            link_transform = kinematic_state.link_transforms.get(link.name)
            if link_transform:
                # Transform link COM to world frame
                link_com_world = pin.math.transform.transform_point(link_transform, link.center_of_mass)
                
                total_mass += link.mass
                weighted_position += link.mass * link_com_world
    
    if total_mass > 0:
        return weighted_position / total_mass
    else:
        return np.zeros(3)


def demonstrate_biped_dynamics(robot):
    """Demonstrate dynamics analysis for biped."""
    print("\n=== Biped Dynamics Analysis ===")
    
    # Standing configuration
    q_stand = np.array([0.0, 0.0, 0.0, 0.0])
    qd_stand = np.array([0.0, 0.0, 0.0, 0.0])
    
    print("Standing configuration analysis:")
    
    # Compute mass matrix
    M = pin.compute_mass_matrix(robot, q_stand)
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix diagonal: {np.diag(M)}")
    
    # Compute gravity torques
    gravity_torques = pin.compute_gravity_vector(robot, q_stand)
    print(f"Gravity torques: {gravity_torques}")
    
    # Test forward dynamics with different torques
    test_torques = [
        np.array([0.0, 0.0, 0.0, 0.0]),  # No torque
        np.array([10.0, 0.0, 0.0, 0.0]),  # Left hip torque
        np.array([0.0, 5.0, 0.0, 0.0]),   # Left knee torque
        np.array([5.0, 2.0, 5.0, 2.0]),  # Balanced torques
    ]
    
    for i, tau in enumerate(test_torques):
        qdd = pin.compute_forward_dynamics(robot, q_stand, qd_stand, tau)
        print(f"Torque {i+1}: {tau} -> Acceleration: {qdd}")


def demonstrate_walking_gait(robot):
    """Demonstrate a simple walking gait pattern."""
    print("\n=== Walking Gait Demonstration ===")
    
    # Simple walking pattern (sinusoidal)
    n_steps = 50
    t = np.linspace(0, 2*np.pi, n_steps)
    
    trajectory = []
    for time in t:
        # Hip oscillation (out of phase for left/right)
        left_hip = 0.3 * np.sin(time)
        right_hip = 0.3 * np.sin(time + np.pi)
        
        # Knee flexion (always positive, synchronized with hip)
        left_knee = 0.5 * (1 + np.cos(time)) * 0.5
        right_knee = 0.5 * (1 + np.cos(time + np.pi)) * 0.5
        
        config = np.array([left_hip, left_knee, right_hip, right_knee])
        trajectory.append(config)
    
    # Analyze trajectory
    print("Walking trajectory analysis:")
    print(f"Number of steps: {len(trajectory)}")
    
    # Compute foot trajectories
    left_foot_trajectory = []
    right_foot_trajectory = []
    
    for config in trajectory:
        try:
            left_pos = pin.get_link_position(robot, config, "left_foot")
            right_pos = pin.get_link_position(robot, config, "right_foot")
            left_foot_trajectory.append(left_pos)
            right_foot_trajectory.append(right_pos)
        except:
            continue
    
    if left_foot_trajectory:
        left_foot_trajectory = np.array(left_foot_trajectory)
        right_foot_trajectory = np.array(right_foot_trajectory)
        
        print(f"Left foot height range: [{left_foot_trajectory[:, 2].min():.3f}, {left_foot_trajectory[:, 2].max():.3f}]")
        print(f"Right foot height range: [{right_foot_trajectory[:, 2].min():.3f}, {right_foot_trajectory[:, 2].max():.3f}]")
        print(f"Step length (left): {left_foot_trajectory[:, 0].max() - left_foot_trajectory[:, 0].min():.3f} m")
        print(f"Step length (right): {right_foot_trajectory[:, 0].max() - right_foot_trajectory[:, 0].min():.3f} m")
    
    return trajectory


def visualize_biped(robot):
    """Visualize the biped robot."""
    print("\n=== Biped Visualization ===")
    
    if not hasattr(pin, 'plot_robot_3d'):
        print("Visualization not available. Install matplotlib to see robot plots.")
        return
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Different poses
    poses = {
        "Standing": np.array([0.0, 0.0, 0.0, 0.0]),
        "Left Step": np.array([0.3, 0.5, -0.2, 0.3]),
        "Squat": np.array([0.5, 1.0, 0.5, 1.0]),
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (pose_name, config) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        try:
            pin.plot_robot_3d(robot, config, ax=ax, show_labels=False)
            ax.set_title(f"Biped: {pose_name}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-1, 0.5)
        except Exception as e:
            print(f"Visualization error for {pose_name}: {e}")
    
    plt.tight_layout()
    plt.savefig("examples/biped_poses.png", dpi=150, bbox_inches='tight')
    print("Saved biped visualization to 'examples/biped_poses.png'")


def analyze_biped_stability(robot):
    """Analyze static stability of different poses."""
    print("\n=== Stability Analysis ===")
    
    poses = {
        "standing": np.array([0.0, 0.0, 0.0, 0.0]),
        "left_lean": np.array([0.2, 0.0, -0.1, 0.0]),
        "right_lean": np.array([-0.1, 0.0, 0.2, 0.0]),
        "forward_lean": np.array([0.1, 0.0, 0.1, 0.0]),
    }
    
    for pose_name, config in poses.items():
        print(f"\nAnalyzing pose: {pose_name}")
        
        try:
            # Compute center of mass
            com = compute_center_of_mass(robot, config)
            
            # Compute foot positions
            left_foot = pin.get_link_position(robot, config, "left_foot")
            right_foot = pin.get_link_position(robot, config, "right_foot")
            
            # Support polygon (simplified as line between feet)
            support_center_x = (left_foot[0] + right_foot[0]) / 2
            support_width = abs(left_foot[0] - right_foot[0])
            
            # Check if COM is within support polygon
            com_offset = abs(com[0] - support_center_x)
            stable = com_offset < support_width / 2
            
            print(f"  COM position: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
            print(f"  Support center: {support_center_x:.3f}")
            print(f"  Support width: {support_width:.3f}")
            print(f"  COM offset: {com_offset:.3f}")
            print(f"  Statically stable: {stable}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main demonstration function."""
    print("py-pinocchio Legged Robot Example")
    print("=" * 50)
    
    # Create biped robot
    robot = create_simple_biped()
    
    # Run demonstrations
    demonstrate_biped_kinematics(robot)
    demonstrate_biped_dynamics(robot)
    walking_trajectory = demonstrate_walking_gait(robot)
    analyze_biped_stability(robot)
    visualize_biped(robot)
    
    print("\n" + "=" * 50)
    print("Legged robot example completed!")
    print("\nKey concepts demonstrated:")
    print("- Multi-link kinematic chains")
    print("- Walking gait patterns")
    print("- Center of mass computation")
    print("- Static stability analysis")
    print("- Complex robot dynamics")
    print("- Biped robot modeling")
    
    print(f"\nRobot specifications:")
    print(f"- Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"- Degrees of freedom: {robot.num_dof}")
    print(f"- Links: {robot.num_links}")
    print(f"- Joints: {robot.num_joints}")


if __name__ == "__main__":
    main()
