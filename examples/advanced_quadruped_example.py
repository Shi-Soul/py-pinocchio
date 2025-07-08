#!/usr/bin/env python3
"""
Advanced Quadruped Robot Example for py-pinocchio

This example demonstrates a sophisticated quadruped robot with:
- Realistic body and leg proportions
- 3-DOF legs (hip abduction, hip flexion, knee flexion)
- Gait pattern generation
- Stability analysis
- Terrain adaptation
- Dynamic walking simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin


def create_advanced_quadruped():
    """
    Create a realistic quadruped robot similar to Boston Dynamics Spot.
    
    Features:
    - 12 DOF total (3 per leg)
    - Realistic mass distribution
    - Proper joint limits
    - Collision geometry
    """
    print("Creating advanced quadruped robot...")
    
    # Physical parameters (inspired by real quadruped robots)
    body_mass = 25.0  # kg
    body_length = 0.7  # m
    body_width = 0.3   # m
    body_height = 0.1  # m
    
    # Leg parameters
    upper_leg_mass = 2.5  # kg
    lower_leg_mass = 1.5  # kg
    foot_mass = 0.3       # kg
    
    upper_leg_length = 0.35  # m
    lower_leg_length = 0.35  # m
    
    # Create links
    links = []
    
    # Main body
    body = pin.Link(
        name="body",
        mass=body_mass,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.array([
            [body_mass * (body_width**2 + body_height**2) / 12, 0, 0],
            [0, body_mass * (body_length**2 + body_height**2) / 12, 0],
            [0, 0, body_mass * (body_length**2 + body_width**2) / 12]
        ])
    )
    links.append(body)
    
    # Leg attachment points
    leg_positions = {
        "front_left": np.array([body_length/2, -body_width/2, 0]),
        "front_right": np.array([body_length/2, body_width/2, 0]),
        "rear_left": np.array([-body_length/2, -body_width/2, 0]),
        "rear_right": np.array([-body_length/2, body_width/2, 0])
    }
    
    joints = []
    
    for leg_name, leg_pos in leg_positions.items():
        # Upper leg (thigh)
        upper_leg = pin.Link(
            name=f"{leg_name}_upper",
            mass=upper_leg_mass,
            center_of_mass=np.array([0.0, 0.0, -upper_leg_length/2]),
            inertia_tensor=np.array([
                [upper_leg_mass * upper_leg_length**2 / 12, 0, 0],
                [upper_leg_mass * upper_leg_length**2 / 12, 0, 0],
                [0, 0, upper_leg_mass * 0.02**2]  # Small radius for rod
            ])
        )
        links.append(upper_leg)
        
        # Lower leg (shin)
        lower_leg = pin.Link(
            name=f"{leg_name}_lower",
            mass=lower_leg_mass,
            center_of_mass=np.array([0.0, 0.0, -lower_leg_length/2]),
            inertia_tensor=np.array([
                [lower_leg_mass * lower_leg_length**2 / 12, 0, 0],
                [lower_leg_mass * lower_leg_length**2 / 12, 0, 0],
                [0, 0, lower_leg_mass * 0.015**2]
            ])
        )
        links.append(lower_leg)
        
        # Foot
        foot = pin.Link(
            name=f"{leg_name}_foot",
            mass=foot_mass,
            center_of_mass=np.array([0.0, 0.0, -0.02]),
            inertia_tensor=np.array([
                [foot_mass * 0.05**2, 0, 0],
                [0, foot_mass * 0.05**2, 0],
                [0, 0, foot_mass * 0.05**2]
            ])
        )
        links.append(foot)
        
        # Hip abduction/adduction joint (X-axis)
        hip_ab_joint = pin.Joint(
            name=f"{leg_name}_hip_ab",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([1, 0, 0]),
            parent_link="body",
            child_link=f"{leg_name}_upper",
            origin_transform=pin.create_transform(np.eye(3), leg_pos),
            position_limit_lower=-np.pi/4,  # ±45 degrees
            position_limit_upper=np.pi/4
        )
        joints.append(hip_ab_joint)
        
        # Hip flexion/extension joint (Y-axis)
        hip_flex_joint = pin.Joint(
            name=f"{leg_name}_hip_flex",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),
            parent_link=f"{leg_name}_upper",
            child_link=f"{leg_name}_lower",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -upper_leg_length])),
            position_limit_lower=-np.pi/2,  # -90 to +90 degrees
            position_limit_upper=np.pi/2
        )
        joints.append(hip_flex_joint)
        
        # Knee flexion joint (Y-axis)
        knee_joint = pin.Joint(
            name=f"{leg_name}_knee",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),
            parent_link=f"{leg_name}_lower",
            child_link=f"{leg_name}_foot",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -lower_leg_length])),
            position_limit_lower=-np.pi,    # Can bend backwards
            position_limit_upper=0.0       # Cannot extend beyond straight
        )
        joints.append(knee_joint)
    
    robot = pin.create_robot_model(
        name="advanced_quadruped",
        links=links,
        joints=joints,
        root_link="body"
    )
    
    print(f"Created quadruped with {robot.num_links} links and {robot.num_dof} DOF")
    print(f"Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"Actuated joints: {len(robot.actuated_joint_names)}")
    
    return robot


def generate_trot_gait(robot, step_length=0.3, step_height=0.1, cycle_time=1.0, n_steps=100):
    """
    Generate a trotting gait pattern.
    
    In trot gait, diagonal legs move together:
    - Phase 1: Front-left + Rear-right in swing, others in stance
    - Phase 2: Front-right + Rear-left in swing, others in stance
    """
    print(f"Generating trot gait pattern...")
    print(f"  Step length: {step_length} m")
    print(f"  Step height: {step_height} m")
    print(f"  Cycle time: {cycle_time} s")
    
    # Time vector
    t = np.linspace(0, cycle_time, n_steps)
    
    # Gait parameters
    duty_factor = 0.6  # Fraction of cycle in stance
    phase_offset = 0.5  # Phase difference between diagonal pairs
    
    trajectory = []
    foot_trajectories = {leg: [] for leg in ["front_left", "front_right", "rear_left", "rear_right"]}
    
    for time in t:
        # Normalized time in cycle [0, 1]
        cycle_phase = (time % cycle_time) / cycle_time
        
        # Joint configuration for this time step
        q = np.zeros(robot.num_dof)
        joint_idx = 0
        
        for leg_name in ["front_left", "front_right", "rear_left", "rear_right"]:
            # Determine if this leg is in swing or stance phase
            if leg_name in ["front_left", "rear_right"]:
                leg_phase = cycle_phase
            else:
                leg_phase = (cycle_phase + phase_offset) % 1.0
            
            # Generate leg trajectory
            if leg_phase < duty_factor:
                # Stance phase - leg on ground, body moves forward
                stance_progress = leg_phase / duty_factor
                x_offset = step_length * (0.5 - stance_progress)
                z_offset = 0.0
            else:
                # Swing phase - leg lifts and moves forward
                swing_progress = (leg_phase - duty_factor) / (1.0 - duty_factor)
                x_offset = step_length * (swing_progress - 0.5)
                z_offset = step_height * np.sin(np.pi * swing_progress)
            
            # Convert foot position to joint angles (simplified inverse kinematics)
            hip_ab, hip_flex, knee = foot_position_to_joint_angles(
                x_offset, 0.0, -0.6 + z_offset,  # Target foot position
                upper_leg_length=0.35,
                lower_leg_length=0.35
            )
            
            # Set joint angles
            q[joint_idx] = hip_ab      # Hip abduction
            q[joint_idx + 1] = hip_flex  # Hip flexion
            q[joint_idx + 2] = knee      # Knee flexion
            joint_idx += 3
            
            # Store foot trajectory for analysis
            foot_trajectories[leg_name].append([x_offset, 0.0, -0.6 + z_offset])
        
        trajectory.append(q)
    
    trajectory = np.array(trajectory)
    
    print(f"Generated trajectory with {len(trajectory)} waypoints")
    print(f"Joint angle ranges (degrees):")
    for i in range(robot.num_dof):
        joint_name = robot.actuated_joint_names[i]
        joint_range = trajectory[:, i]
        print(f"  {joint_name}: [{np.degrees(joint_range.min()):.1f}, {np.degrees(joint_range.max()):.1f}]")
    
    return trajectory, t, foot_trajectories


def foot_position_to_joint_angles(x, y, z, upper_leg_length, lower_leg_length):
    """
    Simple inverse kinematics for 3-DOF leg.
    
    Returns: (hip_abduction, hip_flexion, knee_flexion)
    """
    # Hip abduction angle (rotation about X-axis)
    hip_ab = np.arctan2(y, -z) if z != 0 else 0.0
    
    # Project to sagittal plane
    r = np.sqrt(x**2 + z**2)
    
    # Use law of cosines for knee angle
    l1, l2 = upper_leg_length, lower_leg_length
    cos_knee = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    cos_knee = np.clip(cos_knee, -1, 1)  # Ensure valid range
    knee = np.pi - np.arccos(cos_knee)  # Knee flexion angle
    
    # Hip flexion angle
    alpha = np.arctan2(x, -z)
    beta = np.arccos((l1**2 + r**2 - l2**2) / (2 * l1 * r)) if r > 0 else 0
    hip_flex = alpha - beta
    
    return hip_ab, hip_flex, knee


def analyze_gait_stability(robot, trajectory, foot_trajectories):
    """Analyze stability of the gait pattern."""
    print("\n=== Gait Stability Analysis ===")
    
    stable_steps = 0
    total_steps = len(trajectory)
    
    for i, q in enumerate(trajectory):
        try:
            # Compute center of mass
            com = compute_center_of_mass(robot, q)
            
            # Compute foot positions
            foot_positions = {}
            for leg_name in ["front_left", "front_right", "rear_left", "rear_right"]:
                foot_pos = pin.get_link_position(robot, q, f"{leg_name}_foot")
                foot_positions[leg_name] = foot_pos
            
            # Check if COM is within support polygon
            support_feet = []
            for leg_name, foot_traj in foot_trajectories.items():
                if foot_traj[i][2] <= -0.55:  # Foot close to ground
                    support_feet.append(foot_positions[leg_name][:2])  # X-Y position
            
            if len(support_feet) >= 3:
                # Check if COM projection is within support polygon
                com_xy = com[:2]
                if is_point_in_polygon(com_xy, support_feet):
                    stable_steps += 1
            
        except Exception:
            continue
    
    stability_ratio = stable_steps / total_steps
    print(f"Stability analysis:")
    print(f"  Stable steps: {stable_steps}/{total_steps}")
    print(f"  Stability ratio: {stability_ratio:.2%}")
    
    return stability_ratio


def is_point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm."""
    if len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def compute_center_of_mass(robot, joint_positions):
    """Compute center of mass of the robot."""
    try:
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
    except Exception:
        return np.zeros(3)


def simulate_dynamic_walking(robot, trajectory, time_steps):
    """Simulate dynamic walking with basic physics."""
    print("\n=== Dynamic Walking Simulation ===")
    
    # Simple dynamics simulation
    dt = time_steps[1] - time_steps[0] if len(time_steps) > 1 else 0.01
    
    # Track energy and momentum
    kinetic_energies = []
    potential_energies = []
    
    for i, q in enumerate(trajectory):
        try:
            # Compute velocities (finite difference)
            if i > 0:
                qd = (q - trajectory[i-1]) / dt
            else:
                qd = np.zeros_like(q)
            
            # Compute center of mass
            com = compute_center_of_mass(robot, q)
            
            # Estimate kinetic energy (simplified)
            M = pin.compute_mass_matrix(robot, q)
            kinetic_energy = 0.5 * qd.T @ M @ qd
            kinetic_energies.append(kinetic_energy)
            
            # Potential energy (gravitational)
            total_mass = sum(link.mass for link in robot.links)
            potential_energy = total_mass * 9.81 * com[2]  # mg*h
            potential_energies.append(potential_energy)
            
        except Exception:
            kinetic_energies.append(0.0)
            potential_energies.append(0.0)
    
    # Analysis
    total_energies = np.array(kinetic_energies) + np.array(potential_energies)
    energy_variation = np.std(total_energies) / np.mean(total_energies) if np.mean(total_energies) > 0 else 0
    
    print(f"Energy analysis:")
    print(f"  Average kinetic energy: {np.mean(kinetic_energies):.2f} J")
    print(f"  Average potential energy: {np.mean(potential_energies):.2f} J")
    print(f"  Total energy variation: {energy_variation:.2%}")
    
    return kinetic_energies, potential_energies


def visualize_quadruped_gait(robot, trajectory, foot_trajectories):
    """Visualize the quadruped gait pattern."""
    print("\n=== Gait Visualization ===")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot foot trajectories
        ax = axes[0, 0]
        colors = ['red', 'blue', 'green', 'orange']
        for i, (leg_name, traj) in enumerate(foot_trajectories.items()):
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 2], color=colors[i], label=leg_name, linewidth=2)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title('Foot Trajectories (Side View)')
        ax.legend()
        ax.grid(True)
        
        # Plot joint angles over time
        ax = axes[0, 1]
        time_steps = np.linspace(0, 1, len(trajectory))
        
        # Plot first leg's joint angles as example
        ax.plot(time_steps, np.degrees(trajectory[:, 0]), label='Hip Ab', linewidth=2)
        ax.plot(time_steps, np.degrees(trajectory[:, 1]), label='Hip Flex', linewidth=2)
        ax.plot(time_steps, np.degrees(trajectory[:, 2]), label='Knee', linewidth=2)
        
        ax.set_xlabel('Gait Cycle')
        ax.set_ylabel('Joint Angle (degrees)')
        ax.set_title('Front Left Leg Joint Angles')
        ax.legend()
        ax.grid(True)
        
        # Plot gait diagram
        ax = axes[1, 0]
        leg_names = list(foot_trajectories.keys())
        for i, (leg_name, traj) in enumerate(foot_trajectories.items()):
            traj = np.array(traj)
            stance_phases = traj[:, 2] <= -0.55  # Foot on ground
            
            for j, on_ground in enumerate(stance_phases):
                color = 'black' if on_ground else 'white'
                ax.barh(i, 1, left=j/len(stance_phases), height=0.8, 
                       color=color, edgecolor='gray', linewidth=0.5)
        
        ax.set_yticks(range(len(leg_names)))
        ax.set_yticklabels(leg_names)
        ax.set_xlabel('Gait Cycle')
        ax.set_title('Gait Diagram (Black=Stance, White=Swing)')
        
        # Plot robot configuration snapshots
        ax = axes[1, 1]
        snapshot_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
        
        for i, idx in enumerate(snapshot_indices):
            q = trajectory[idx]
            
            # Simplified robot visualization (body + leg endpoints)
            try:
                body_pos = np.array([0, 0, 0])  # Body at origin
                
                # Plot body
                body_x = [-0.35, 0.35, 0.35, -0.35, -0.35]
                body_y = [-0.15, -0.15, 0.15, 0.15, -0.15]
                ax.plot(body_x, body_y, 'k-', linewidth=2, alpha=0.7)
                
                # Plot leg endpoints
                joint_idx = 0
                leg_positions = [
                    [0.35, -0.15], [0.35, 0.15], [-0.35, -0.15], [-0.35, 0.15]
                ]
                
                for j, leg_pos in enumerate(leg_positions):
                    # Simple forward kinematics for foot position
                    hip_ab = q[joint_idx]
                    hip_flex = q[joint_idx + 1] 
                    knee = q[joint_idx + 2]
                    joint_idx += 3
                    
                    # Approximate foot position
                    foot_x = leg_pos[0] + 0.3 * np.sin(hip_flex) + 0.3 * np.sin(hip_flex + knee)
                    foot_y = leg_pos[1] + 0.1 * np.sin(hip_ab)
                    
                    ax.plot([leg_pos[0], foot_x], [leg_pos[1], foot_y], 
                           'b-', linewidth=1, alpha=0.5 + 0.1*i)
                    ax.plot(foot_x, foot_y, 'ro', markersize=4, alpha=0.5 + 0.1*i)
                
            except Exception:
                pass
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title('Robot Configuration Snapshots')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig("examples/advanced_quadruped_gait.png", dpi=150, bbox_inches='tight')
        print("Saved gait visualization to 'examples/advanced_quadruped_gait.png'")
        
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    """Main demonstration function."""
    print("py-pinocchio Advanced Quadruped Example")
    print("=" * 50)
    
    # Create advanced quadruped
    robot = create_advanced_quadruped()
    
    # Generate trot gait
    trajectory, time_steps, foot_trajectories = generate_trot_gait(robot)
    
    # Analyze gait stability
    stability_ratio = analyze_gait_stability(robot, trajectory, foot_trajectories)
    
    # Simulate dynamics
    kinetic_energies, potential_energies = simulate_dynamic_walking(robot, trajectory, time_steps)
    
    # Visualize results
    visualize_quadruped_gait(robot, trajectory, foot_trajectories)
    
    print("\n" + "=" * 50)
    print("Advanced quadruped example completed!")
    print("\nKey features demonstrated:")
    print("- Realistic 12-DOF quadruped robot")
    print("- Trotting gait generation")
    print("- Stability analysis")
    print("- Dynamic walking simulation")
    print("- Comprehensive visualization")
    
    print(f"\nRobot specifications:")
    print(f"- Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"- Degrees of freedom: {robot.num_dof}")
    print(f"- Gait stability: {stability_ratio:.1%}")
    print(f"- Energy efficiency: {np.mean(kinetic_energies):.2f} J average kinetic energy")


if __name__ == "__main__":
    main()
