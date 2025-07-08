#!/usr/bin/env python3
"""
Industrial Manipulator Example for py-pinocchio

This example demonstrates a sophisticated industrial robot arm with:
- 6-DOF articulated configuration
- Realistic industrial robot proportions
- Advanced trajectory planning
- Collision avoidance concepts
- Pick-and-place operations
- Workspace optimization
- Tool center point (TCP) control
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin


def create_industrial_manipulator():
    """
    Create a 6-DOF industrial manipulator similar to ABB IRB or KUKA robots.
    
    Features:
    - Anthropomorphic configuration
    - Realistic link lengths and masses
    - Industrial-grade joint limits
    - Tool mounting flange
    """
    print("Creating industrial manipulator...")
    
    # Robot parameters (based on medium-payload industrial robots)
    base_height = 0.4      # m
    shoulder_offset = 0.15  # m
    upper_arm_length = 0.7  # m
    forearm_length = 0.6    # m
    wrist_length = 0.2      # m
    
    # Masses (realistic for industrial robot)
    base_mass = 50.0        # kg
    shoulder_mass = 30.0    # kg
    upper_arm_mass = 25.0   # kg
    forearm_mass = 15.0     # kg
    wrist1_mass = 8.0       # kg
    wrist2_mass = 5.0       # kg
    wrist3_mass = 3.0       # kg
    
    links = []
    
    # Base link
    base = pin.Link(
        name="base_link",
        mass=base_mass,
        center_of_mass=np.array([0.0, 0.0, base_height/2]),
        inertia_tensor=np.array([
            [base_mass * 0.2**2, 0, 0],
            [0, base_mass * 0.2**2, 0],
            [0, 0, base_mass * 0.15**2]
        ])
    )
    links.append(base)
    
    # Shoulder link
    shoulder = pin.Link(
        name="shoulder_link",
        mass=shoulder_mass,
        center_of_mass=np.array([0.0, 0.0, shoulder_offset/2]),
        inertia_tensor=np.array([
            [shoulder_mass * 0.1**2, 0, 0],
            [0, shoulder_mass * 0.1**2, 0],
            [0, 0, shoulder_mass * 0.08**2]
        ])
    )
    links.append(shoulder)
    
    # Upper arm link
    upper_arm = pin.Link(
        name="upper_arm_link",
        mass=upper_arm_mass,
        center_of_mass=np.array([upper_arm_length/2, 0.0, 0.0]),
        inertia_tensor=np.array([
            [upper_arm_mass * 0.05**2, 0, 0],
            [0, upper_arm_mass * upper_arm_length**2 / 12, 0],
            [0, 0, upper_arm_mass * upper_arm_length**2 / 12]
        ])
    )
    links.append(upper_arm)
    
    # Forearm link
    forearm = pin.Link(
        name="forearm_link",
        mass=forearm_mass,
        center_of_mass=np.array([forearm_length/2, 0.0, 0.0]),
        inertia_tensor=np.array([
            [forearm_mass * 0.04**2, 0, 0],
            [0, forearm_mass * forearm_length**2 / 12, 0],
            [0, 0, forearm_mass * forearm_length**2 / 12]
        ])
    )
    links.append(forearm)
    
    # Wrist links
    wrist1 = pin.Link(
        name="wrist_1_link",
        mass=wrist1_mass,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.array([
            [wrist1_mass * 0.03**2, 0, 0],
            [0, wrist1_mass * 0.03**2, 0],
            [0, 0, wrist1_mass * 0.02**2]
        ])
    )
    links.append(wrist1)
    
    wrist2 = pin.Link(
        name="wrist_2_link",
        mass=wrist2_mass,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.array([
            [wrist2_mass * 0.025**2, 0, 0],
            [0, wrist2_mass * 0.025**2, 0],
            [0, 0, wrist2_mass * 0.02**2]
        ])
    )
    links.append(wrist2)
    
    wrist3 = pin.Link(
        name="wrist_3_link",
        mass=wrist3_mass,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.array([
            [wrist3_mass * 0.02**2, 0, 0],
            [0, wrist3_mass * 0.02**2, 0],
            [0, 0, wrist3_mass * 0.015**2]
        ])
    )
    links.append(wrist3)
    
    # Tool flange
    tool_flange = pin.Link(
        name="tool_flange",
        mass=1.0,
        center_of_mass=np.array([0.0, 0.0, 0.05]),
        inertia_tensor=np.array([
            [0.001, 0, 0],
            [0, 0.001, 0],
            [0, 0, 0.001]
        ])
    )
    links.append(tool_flange)
    
    # Create joints
    joints = []
    
    # Joint 1: Base rotation
    joint1 = pin.Joint(
        name="joint_1",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),
        parent_link="base_link",
        child_link="shoulder_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, base_height])),
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    joints.append(joint1)
    
    # Joint 2: Shoulder pitch
    joint2 = pin.Joint(
        name="joint_2",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="shoulder_link",
        child_link="upper_arm_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, shoulder_offset])),
        position_limit_lower=-np.pi/2,
        position_limit_upper=np.pi/2
    )
    joints.append(joint2)
    
    # Joint 3: Elbow pitch
    joint3 = pin.Joint(
        name="joint_3",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="upper_arm_link",
        child_link="forearm_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([upper_arm_length, 0, 0])),
        position_limit_lower=-np.pi,
        position_limit_upper=0
    )
    joints.append(joint3)
    
    # Joint 4: Wrist 1 (roll)
    joint4 = pin.Joint(
        name="joint_4",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([1, 0, 0]),
        parent_link="forearm_link",
        child_link="wrist_1_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([forearm_length, 0, 0])),
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    joints.append(joint4)
    
    # Joint 5: Wrist 2 (pitch)
    joint5 = pin.Joint(
        name="joint_5",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),
        parent_link="wrist_1_link",
        child_link="wrist_2_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, 0])),
        position_limit_lower=-np.pi/2,
        position_limit_upper=np.pi/2
    )
    joints.append(joint5)
    
    # Joint 6: Wrist 3 (yaw)
    joint6 = pin.Joint(
        name="joint_6",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),
        parent_link="wrist_2_link",
        child_link="wrist_3_link",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, 0])),
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    joints.append(joint6)
    
    # Tool flange connection (fixed)
    tool_joint = pin.Joint(
        name="tool_joint",
        joint_type=pin.JointType.FIXED,
        parent_link="wrist_3_link",
        child_link="tool_flange",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, wrist_length]))
    )
    joints.append(tool_joint)
    
    robot = pin.create_robot_model(
        name="industrial_manipulator",
        links=links,
        joints=joints,
        root_link="base_link"
    )
    
    print(f"Created manipulator with {robot.num_links} links and {robot.num_dof} DOF")
    print(f"Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"Reach: ~{upper_arm_length + forearm_length + wrist_length:.1f} m")
    
    return robot


def demonstrate_industrial_configurations(robot):
    """Demonstrate typical industrial robot configurations."""
    print("\n=== Industrial Robot Configurations ===")
    
    # Define industrial robot poses
    configurations = {
        "home": np.array([0, 0, 0, 0, 0, 0]),
        "ready": np.array([0, -np.pi/4, -np.pi/2, 0, -np.pi/4, 0]),
        "pick_high": np.array([np.pi/4, -np.pi/6, -np.pi/3, 0, -np.pi/2, 0]),
        "pick_low": np.array([0, np.pi/6, -np.pi/4, 0, np.pi/3, 0]),
        "place_side": np.array([np.pi/2, -np.pi/4, -np.pi/2, np.pi/2, 0, 0]),
        "maintenance": np.array([0, -np.pi/2, -np.pi/2, 0, 0, 0]),
    }
    
    for config_name, q in configurations.items():
        print(f"\nConfiguration: {config_name}")
        print(f"  Joint angles (deg): {np.degrees(q)}")
        
        try:
            # Check joint limits
            if not robot.is_valid_joint_configuration(q):
                print(f"  Warning: Configuration violates joint limits")
                continue
            
            # Compute tool center point (TCP) position
            tcp_pos = pin.get_link_position(robot, q, "tool_flange")
            tcp_rot = pin.get_link_orientation(robot, q, "tool_flange")
            
            print(f"  TCP position: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
            print(f"  Distance from base: {np.linalg.norm(tcp_pos):.3f} m")
            
            # Compute manipulability
            jacobian = pin.compute_geometric_jacobian(robot, q, "tool_flange")
            manipulability = compute_manipulability(jacobian)
            condition_number = np.linalg.cond(jacobian)
            
            print(f"  Manipulability: {manipulability:.6f}")
            print(f"  Condition number: {condition_number:.2f}")
            
            # Check for singularities
            is_singular = condition_number > 1000
            print(f"  Near singularity: {is_singular}")
            
        except Exception as e:
            print(f"  Error: {e}")


def compute_manipulability(jacobian):
    """Compute manipulability measure."""
    try:
        # Use Yoshikawa's manipulability measure
        J = jacobian[3:6, :]  # Linear velocity part
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        return manipulability
    except:
        return 0.0


def plan_pick_and_place_trajectory(robot, pick_pos, place_pos, approach_height=0.1):
    """Plan a pick-and-place trajectory."""
    print(f"\n=== Pick-and-Place Trajectory Planning ===")
    print(f"Pick position: {pick_pos}")
    print(f"Place position: {place_pos}")
    print(f"Approach height: {approach_height} m")
    
    # Define waypoints
    pick_approach = pick_pos + np.array([0, 0, approach_height])
    place_approach = place_pos + np.array([0, 0, approach_height])
    
    waypoints = [
        ("home", np.array([0, -np.pi/4, -np.pi/2, 0, -np.pi/4, 0])),
        ("pick_approach", None),  # Will be computed via IK
        ("pick", None),
        ("pick_retreat", None),
        ("place_approach", None),
        ("place", None),
        ("place_retreat", None),
        ("home", np.array([0, -np.pi/4, -np.pi/2, 0, -np.pi/4, 0])),
    ]
    
    trajectory = []
    
    for waypoint_name, q_target in waypoints:
        if q_target is not None:
            # Use provided joint configuration
            trajectory.append((waypoint_name, q_target))
        else:
            # Compute via inverse kinematics
            if "pick_approach" in waypoint_name:
                target_pos = pick_approach
            elif "pick" in waypoint_name and "approach" not in waypoint_name:
                target_pos = pick_pos
            elif "pick_retreat" in waypoint_name:
                target_pos = pick_approach
            elif "place_approach" in waypoint_name:
                target_pos = place_approach
            elif "place" in waypoint_name and "approach" not in waypoint_name:
                target_pos = place_pos
            elif "place_retreat" in waypoint_name:
                target_pos = place_approach
            else:
                continue
            
            # Simple inverse kinematics (geometric approach for this robot)
            q_ik = simple_inverse_kinematics(robot, target_pos)
            
            if q_ik is not None:
                trajectory.append((waypoint_name, q_ik))
                print(f"  {waypoint_name}: IK solution found")
            else:
                print(f"  {waypoint_name}: IK failed")
    
    print(f"Generated trajectory with {len(trajectory)} waypoints")
    return trajectory


def simple_inverse_kinematics(robot, target_position, max_iterations=100):
    """
    Simple inverse kinematics using Jacobian pseudoinverse.
    """
    # Initial guess (ready position)
    q = np.array([0, -np.pi/4, -np.pi/2, 0, -np.pi/4, 0])
    target = np.array(target_position)
    
    for iteration in range(max_iterations):
        try:
            # Current TCP position
            current_pos = pin.get_link_position(robot, q, "tool_flange")
            
            # Position error
            error = target - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 1e-3:  # Converged
                return q
            
            # Compute Jacobian (position part only)
            jacobian_full = pin.compute_geometric_jacobian(robot, q, "tool_flange")
            jacobian_pos = jacobian_full[3:6, :]  # Linear velocity part
            
            # Pseudoinverse step
            jacobian_pinv = np.linalg.pinv(jacobian_pos)
            dq = jacobian_pinv @ error
            
            # Update with step size
            step_size = 0.1
            q_new = q + step_size * dq
            
            # Enforce joint limits
            for i, joint_name in enumerate(robot.actuated_joint_names):
                joint = robot.get_joint(joint_name)
                if joint:
                    q_new[i] = np.clip(q_new[i], joint.position_limit_lower, joint.position_limit_upper)
            
            q = q_new
            
        except Exception:
            break
    
    return None  # Failed to converge


def analyze_workspace_coverage(robot, n_samples=5000):
    """Analyze workspace coverage and reachability."""
    print(f"\n=== Workspace Analysis ===")
    print(f"Sampling {n_samples} random configurations...")
    
    reachable_points = []
    manipulability_scores = []
    
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
        
        q = np.array(q)
        
        try:
            # Compute TCP position
            tcp_pos = pin.get_link_position(robot, q, "tool_flange")
            reachable_points.append(tcp_pos)
            
            # Compute manipulability
            jacobian = pin.compute_geometric_jacobian(robot, q, "tool_flange")
            manip = compute_manipulability(jacobian)
            manipulability_scores.append(manip)
            
        except:
            continue
    
    if reachable_points:
        reachable_points = np.array(reachable_points)
        manipulability_scores = np.array(manipulability_scores)
        
        print(f"Successfully sampled {len(reachable_points)} valid configurations")
        print("Workspace bounds:")
        print(f"  X: [{reachable_points[:, 0].min():.2f}, {reachable_points[:, 0].max():.2f}] m")
        print(f"  Y: [{reachable_points[:, 1].min():.2f}, {reachable_points[:, 1].max():.2f}] m")
        print(f"  Z: [{reachable_points[:, 2].min():.2f}, {reachable_points[:, 2].max():.2f}] m")
        
        # Workspace volume (approximate)
        x_range = reachable_points[:, 0].max() - reachable_points[:, 0].min()
        y_range = reachable_points[:, 1].max() - reachable_points[:, 1].min()
        z_range = reachable_points[:, 2].max() - reachable_points[:, 2].min()
        volume_approx = x_range * y_range * z_range
        
        print(f"  Approximate workspace volume: {volume_approx:.2f} m³")
        
        # Reach analysis
        distances = np.linalg.norm(reachable_points, axis=1)
        print(f"  Maximum reach: {distances.max():.2f} m")
        print(f"  Minimum reach: {distances.min():.2f} m")
        print(f"  Average reach: {distances.mean():.2f} m")
        
        # Manipulability analysis
        print(f"  Average manipulability: {manipulability_scores.mean():.6f}")
        print(f"  Max manipulability: {manipulability_scores.max():.6f}")
        print(f"  Min manipulability: {manipulability_scores.min():.6f}")
        
        # Dexterous workspace (high manipulability)
        high_manip_threshold = np.percentile(manipulability_scores, 75)
        dexterous_points = reachable_points[manipulability_scores > high_manip_threshold]
        dexterous_ratio = len(dexterous_points) / len(reachable_points)
        
        print(f"  Dexterous workspace ratio: {dexterous_ratio:.1%}")
        
        return reachable_points, manipulability_scores
    
    return None, None


def main():
    """Main demonstration function."""
    print("py-pinocchio Industrial Manipulator Example")
    print("=" * 50)
    
    # Create industrial manipulator
    robot = create_industrial_manipulator()
    
    # Demonstrate configurations
    demonstrate_industrial_configurations(robot)
    
    # Plan pick-and-place trajectory
    pick_pos = np.array([0.8, 0.3, 0.2])
    place_pos = np.array([0.5, -0.4, 0.4])
    trajectory = plan_pick_and_place_trajectory(robot, pick_pos, place_pos)
    
    # Analyze workspace
    workspace_points, manipulability_scores = analyze_workspace_coverage(robot)
    
    print("\n" + "=" * 50)
    print("Industrial manipulator example completed!")
    print("\nKey features demonstrated:")
    print("- 6-DOF industrial robot configuration")
    print("- Realistic joint limits and dynamics")
    print("- Pick-and-place trajectory planning")
    print("- Inverse kinematics solutions")
    print("- Workspace and manipulability analysis")
    print("- Tool center point (TCP) control")
    
    print(f"\nRobot specifications:")
    print(f"- Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"- Degrees of freedom: {robot.num_dof}")
    print(f"- Maximum reach: ~1.5 m")
    print(f"- Payload capacity: ~10 kg (estimated)")


if __name__ == "__main__":
    main()
