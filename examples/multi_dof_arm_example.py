#!/usr/bin/env python3
"""
Multi-DOF robotic arm example for py-pinocchio

This example demonstrates modeling and control of a 6-DOF robotic arm
similar to industrial manipulators. We'll explore workspace analysis,
inverse kinematics, and trajectory planning.
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin


def create_6dof_arm():
    """
    Create a 6-DOF robotic arm.
    
    Structure similar to a typical industrial robot:
    - Base (fixed)
    - Link 1: Base rotation (Z-axis)
    - Link 2: Shoulder pitch (Y-axis)  
    - Link 3: Elbow pitch (Y-axis)
    - Link 4: Wrist roll (X-axis)
    - Link 5: Wrist pitch (Y-axis)
    - Link 6: Wrist yaw (Z-axis)
    - End-effector
    """
    print("Creating 6-DOF robotic arm...")
    
    # Link parameters (lengths in meters, masses in kg)
    link_params = [
        {"name": "base", "length": 0.0, "mass": 5.0},
        {"name": "link1", "length": 0.3, "mass": 3.0},  # Base to shoulder
        {"name": "link2", "length": 0.4, "mass": 2.5},  # Upper arm
        {"name": "link3", "length": 0.35, "mass": 2.0}, # Forearm
        {"name": "link4", "length": 0.1, "mass": 1.0},  # Wrist link 1
        {"name": "link5", "length": 0.1, "mass": 0.8},  # Wrist link 2
        {"name": "link6", "length": 0.05, "mass": 0.5}, # Wrist link 3
        {"name": "end_effector", "length": 0.1, "mass": 0.3}, # Tool
    ]
    
    # Create links
    links = []
    for i, params in enumerate(link_params):
        if i == 0:  # Base link
            com = np.array([0.0, 0.0, params["length"]/2])
        else:
            com = np.array([params["length"]/2, 0.0, 0.0])
        
        # Simplified inertia (rod-like)
        mass = params["mass"]
        length = params["length"]
        inertia = np.diag([
            mass * length**2 / 12,  # About perpendicular axes
            mass * length**2 / 12,
            mass * length**2 / 12   # About length axis (simplified)
        ])
        
        link = pin.Link(
            name=params["name"],
            mass=mass,
            center_of_mass=com,
            inertia_tensor=inertia
        )
        links.append(link)
    
    # Create joints
    joints = []
    
    # Joint configurations: (parent, child, axis, origin_offset)
    joint_configs = [
        ("base", "link1", [0, 0, 1], [0, 0, 0.3]),      # Base rotation
        ("link1", "link2", [0, 1, 0], [0, 0, 0]),       # Shoulder pitch
        ("link2", "link3", [0, 1, 0], [0.4, 0, 0]),     # Elbow pitch
        ("link3", "link4", [1, 0, 0], [0.35, 0, 0]),    # Wrist roll
        ("link4", "link5", [0, 1, 0], [0.1, 0, 0]),     # Wrist pitch
        ("link5", "link6", [0, 0, 1], [0.1, 0, 0]),     # Wrist yaw
        ("link6", "end_effector", [0, 0, 0], [0.05, 0, 0]), # Fixed end-effector
    ]
    
    for i, (parent, child, axis, offset) in enumerate(joint_configs):
        if i == len(joint_configs) - 1:  # Last joint is fixed
            joint_type = pin.JointType.FIXED
            limits = (0, 0)
        else:
            joint_type = pin.JointType.REVOLUTE
            limits = (-np.pi, np.pi)  # ±180 degrees
        
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=joint_type,
            axis=np.array(axis),
            parent_link=parent,
            child_link=child,
            origin_transform=pin.create_transform(np.eye(3), np.array(offset)),
            position_limit_lower=limits[0],
            position_limit_upper=limits[1]
        )
        joints.append(joint)
    
    # Create robot model
    robot = pin.create_robot_model(
        name="6dof_arm",
        links=links,
        joints=joints,
        root_link="base"
    )
    
    print(f"Created arm with {robot.num_links} links and {robot.num_dof} DOF")
    print(f"Actuated joints: {robot.actuated_joint_names}")
    
    return robot


def demonstrate_arm_kinematics(robot):
    """Demonstrate forward kinematics for various arm configurations."""
    print("\n=== 6-DOF Arm Kinematics Analysis ===")
    
    # Define interesting configurations
    configs = {
        "home": np.zeros(6),  # All joints at zero
        "reach_forward": np.array([0, -np.pi/4, -np.pi/2, 0, np.pi/4, 0]),
        "reach_up": np.array([0, -np.pi/2, 0, 0, -np.pi/2, 0]),
        "reach_side": np.array([np.pi/2, -np.pi/4, -np.pi/3, 0, 0, 0]),
        "folded": np.array([0, np.pi/3, np.pi/2, 0, np.pi/6, 0]),
        "twisted": np.array([np.pi/4, -np.pi/6, -np.pi/4, np.pi/2, np.pi/3, np.pi/4]),
    }
    
    print("Configuration analysis:")
    for config_name, joint_config in configs.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Joint angles (deg): {np.degrees(joint_config)}")
        
        try:
            # End-effector pose
            ee_pos = pin.get_link_position(robot, joint_config, "end_effector")
            ee_rot = pin.get_link_orientation(robot, joint_config, "end_effector")
            
            print(f"  End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"  Distance from base: {np.linalg.norm(ee_pos):.3f} m")
            
            # Compute reach and manipulability
            jacobian = pin.compute_geometric_jacobian(robot, joint_config, "end_effector")
            analysis = pin.algorithms.jacobian.analyze_jacobian_singularities(jacobian)
            
            print(f"  Manipulability: {analysis['manipulability']:.6f}")
            print(f"  Condition number: {analysis['condition_number']:.2f}")
            print(f"  Near singularity: {analysis['is_singular']}")
            
        except Exception as e:
            print(f"  Error: {e}")


def analyze_workspace(robot):
    """Analyze the workspace of the 6-DOF arm."""
    print("\n=== Workspace Analysis ===")
    
    # Sample workspace with random configurations
    n_samples = 5000
    workspace_points = []
    
    print(f"Sampling {n_samples} random configurations...")
    
    for _ in range(n_samples):
        # Generate random joint configuration within limits
        q = []
        for joint_name in robot.actuated_joint_names:
            joint = robot.get_joint(joint_name)
            if joint:
                q_min = joint.position_limit_lower
                q_max = joint.position_limit_upper
                q.append(np.random.uniform(q_min, q_max))
            else:
                q.append(0.0)
        
        try:
            ee_pos = pin.get_link_position(robot, np.array(q), "end_effector")
            workspace_points.append(ee_pos)
        except:
            continue
    
    if workspace_points:
        workspace_points = np.array(workspace_points)
        
        print(f"Successfully sampled {len(workspace_points)} points")
        print("Workspace bounds:")
        print(f"  X: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}] m")
        print(f"  Y: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}] m")
        print(f"  Z: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}] m")
        
        # Compute workspace volume (approximate)
        x_range = workspace_points[:, 0].max() - workspace_points[:, 0].min()
        y_range = workspace_points[:, 1].max() - workspace_points[:, 1].min()
        z_range = workspace_points[:, 2].max() - workspace_points[:, 2].min()
        volume_approx = x_range * y_range * z_range
        
        print(f"  Approximate workspace volume: {volume_approx:.3f} m³")
        
        # Maximum reach
        distances = np.linalg.norm(workspace_points, axis=1)
        print(f"  Maximum reach: {distances.max():.3f} m")
        print(f"  Minimum reach: {distances.min():.3f} m")
        
        return workspace_points
    else:
        print("No valid workspace points found")
        return None


def demonstrate_arm_dynamics(robot):
    """Demonstrate dynamics analysis for the arm."""
    print("\n=== 6-DOF Arm Dynamics Analysis ===")
    
    # Test configuration
    q = np.array([0, -np.pi/4, -np.pi/2, 0, np.pi/4, 0])  # Reach forward
    qd = np.array([0.1, -0.2, 0.3, 0.5, -0.1, 0.4])      # Joint velocities
    
    print("Configuration for dynamics analysis:")
    print(f"  Joint positions (deg): {np.degrees(q)}")
    print(f"  Joint velocities (rad/s): {qd}")
    
    # Compute dynamics quantities
    try:
        mass_matrix = pin.compute_mass_matrix(robot, q)
        coriolis_forces = pin.compute_coriolis_forces(robot, q, qd)
        gravity_forces = pin.compute_gravity_vector(robot, q)
        
        print(f"\nDynamics analysis:")
        print(f"  Mass matrix shape: {mass_matrix.shape}")
        print(f"  Mass matrix diagonal: {np.diag(mass_matrix)}")
        print(f"  Mass matrix condition number: {np.linalg.cond(mass_matrix):.2f}")
        
        print(f"  Coriolis forces: {coriolis_forces}")
        print(f"  Gravity forces: {gravity_forces}")
        
        # Test forward dynamics with different torques
        test_torques = [
            np.zeros(6),  # No torque
            np.array([1, 0, 0, 0, 0, 0]),  # Base joint only
            np.array([0, 5, 0, 0, 0, 0]),  # Shoulder only
            np.array([1, 2, 1, 0.5, 0.2, 0.1]),  # Distributed torques
        ]
        
        print(f"\nForward dynamics tests:")
        for i, tau in enumerate(test_torques):
            qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
            print(f"  Torque {i+1}: max={tau.max():.1f} -> max_accel={qdd.max():.3f} rad/s²")
            
    except Exception as e:
        print(f"Dynamics computation error: {e}")


def simple_inverse_kinematics(robot, target_position, initial_guess=None, max_iterations=100):
    """
    Simple inverse kinematics using Jacobian pseudoinverse.
    
    This is a basic implementation for educational purposes.
    """
    if initial_guess is None:
        q = np.zeros(robot.num_dof)
    else:
        q = np.array(initial_guess)
    
    target = np.array(target_position)
    
    for iteration in range(max_iterations):
        # Current end-effector position
        current_pos = pin.get_link_position(robot, q, "end_effector")
        
        # Position error
        error = target - current_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < 1e-4:  # Converged
            return q, True, iteration
        
        # Compute Jacobian (position part only)
        jacobian_full = pin.compute_geometric_jacobian(robot, q, "end_effector")
        jacobian_pos = jacobian_full[3:6, :]  # Linear velocity part
        
        # Pseudoinverse step
        jacobian_pinv = np.linalg.pinv(jacobian_pos)
        dq = jacobian_pinv @ error
        
        # Update with step size
        step_size = 0.1
        q = q + step_size * dq
        
        # Enforce joint limits
        for i, joint_name in enumerate(robot.actuated_joint_names):
            joint = robot.get_joint(joint_name)
            if joint:
                q[i] = np.clip(q[i], joint.position_limit_lower, joint.position_limit_upper)
    
    return q, False, max_iterations


def demonstrate_inverse_kinematics(robot):
    """Demonstrate inverse kinematics for target reaching."""
    print("\n=== Inverse Kinematics Demonstration ===")
    
    # Define target positions
    targets = [
        np.array([0.5, 0.0, 0.5]),   # Forward reach
        np.array([0.0, 0.5, 0.3]),   # Side reach
        np.array([0.3, 0.3, 0.7]),   # Up and forward
        np.array([0.2, -0.2, 0.2]),  # Down and back
    ]
    
    print("Inverse kinematics tests:")
    
    for i, target in enumerate(targets):
        print(f"\nTarget {i+1}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        
        try:
            # Solve inverse kinematics
            q_solution, converged, iterations = simple_inverse_kinematics(robot, target)
            
            if converged:
                # Verify solution
                achieved_pos = pin.get_link_position(robot, q_solution, "end_effector")
                error = np.linalg.norm(achieved_pos - target)
                
                print(f"  ✓ Converged in {iterations} iterations")
                print(f"  Solution (deg): {np.degrees(q_solution)}")
                print(f"  Achieved position: [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}]")
                print(f"  Position error: {error:.6f} m")
                
                # Check manipulability at solution
                jacobian = pin.compute_geometric_jacobian(robot, q_solution, "end_effector")
                analysis = pin.algorithms.jacobian.analyze_jacobian_singularities(jacobian)
                print(f"  Manipulability: {analysis['manipulability']:.6f}")
                
            else:
                print(f"  ✗ Failed to converge after {iterations} iterations")
                
        except Exception as e:
            print(f"  Error: {e}")


def plan_trajectory(robot, start_config, end_config, duration=5.0, n_points=50):
    """Plan a simple joint-space trajectory."""
    print(f"\n=== Trajectory Planning ===")
    
    print(f"Planning trajectory from start to end configuration...")
    print(f"  Duration: {duration} seconds")
    print(f"  Points: {n_points}")
    
    # Simple linear interpolation in joint space
    t = np.linspace(0, duration, n_points)
    trajectory = []
    
    for time in t:
        # Linear interpolation
        alpha = time / duration
        q = (1 - alpha) * start_config + alpha * end_config
        trajectory.append(q)
    
    # Analyze trajectory
    trajectory = np.array(trajectory)
    
    print(f"Trajectory analysis:")
    print(f"  Joint ranges (deg):")
    for i in range(robot.num_dof):
        joint_traj = trajectory[:, i]
        print(f"    Joint {i+1}: [{np.degrees(joint_traj.min()):.1f}, {np.degrees(joint_traj.max()):.1f}]")
    
    # Compute end-effector trajectory
    ee_trajectory = []
    for q in trajectory:
        try:
            ee_pos = pin.get_link_position(robot, q, "end_effector")
            ee_trajectory.append(ee_pos)
        except:
            continue
    
    if ee_trajectory:
        ee_trajectory = np.array(ee_trajectory)
        path_length = np.sum(np.linalg.norm(np.diff(ee_trajectory, axis=0), axis=1))
        print(f"  End-effector path length: {path_length:.3f} m")
    
    return trajectory, t


def visualize_arm(robot):
    """Visualize the 6-DOF arm in different configurations."""
    print("\n=== 6-DOF Arm Visualization ===")
    
    if not hasattr(pin, 'plot_robot_3d'):
        print("Visualization not available. Install matplotlib to see robot plots.")
        return
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Configurations to visualize
    configs = {
        "Home": np.zeros(6),
        "Reach Forward": np.array([0, -np.pi/4, -np.pi/2, 0, np.pi/4, 0]),
        "Reach Up": np.array([0, -np.pi/2, 0, 0, -np.pi/2, 0]),
        "Folded": np.array([0, np.pi/3, np.pi/2, 0, np.pi/6, 0]),
    }
    
    fig = plt.figure(figsize=(16, 8))
    
    for i, (config_name, config) in enumerate(configs.items()):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        try:
            pin.plot_robot_3d(robot, config, ax=ax, show_labels=False)
            ax.set_title(f"6-DOF Arm: {config_name}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, 1.5)
        except Exception as e:
            print(f"Visualization error for {config_name}: {e}")
    
    plt.tight_layout()
    plt.savefig("examples/6dof_arm_configurations.png", dpi=150, bbox_inches='tight')
    print("Saved arm visualization to 'examples/6dof_arm_configurations.png'")


def main():
    """Main demonstration function."""
    print("py-pinocchio Multi-DOF Arm Example")
    print("=" * 50)
    
    # Create 6-DOF arm
    robot = create_6dof_arm()
    
    # Run demonstrations
    demonstrate_arm_kinematics(robot)
    workspace_points = analyze_workspace(robot)
    demonstrate_arm_dynamics(robot)
    demonstrate_inverse_kinematics(robot)
    
    # Trajectory planning
    start_config = np.zeros(6)
    end_config = np.array([np.pi/4, -np.pi/3, -np.pi/2, np.pi/6, np.pi/4, -np.pi/6])
    trajectory, time_steps = plan_trajectory(robot, start_config, end_config)
    
    # Visualization
    visualize_arm(robot)
    
    print("\n" + "=" * 50)
    print("Multi-DOF arm example completed!")
    print("\nKey concepts demonstrated:")
    print("- 6-DOF robotic arm modeling")
    print("- Complex workspace analysis")
    print("- Jacobian-based inverse kinematics")
    print("- Manipulability analysis")
    print("- Trajectory planning")
    print("- High-DOF dynamics computation")
    
    print(f"\nRobot specifications:")
    print(f"- Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"- Degrees of freedom: {robot.num_dof}")
    print(f"- Maximum reach: ~{sum([0.4, 0.35, 0.1, 0.1, 0.05, 0.1]):.2f} m")
    print(f"- Workspace volume: ~{1.5:.1f} m³ (estimated)")


if __name__ == "__main__":
    main()
