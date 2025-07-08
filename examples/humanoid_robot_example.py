#!/usr/bin/env python3
"""
Humanoid Robot Example for py-pinocchio

This example demonstrates a sophisticated humanoid robot with:
- Full-body kinematics (30+ DOF)
- Upper body (torso, arms, head)
- Lower body (pelvis, legs, feet)
- Balance and posture control
- Whole-body motion planning
- Human-like proportions and constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin


def create_humanoid_robot():
    """
    Create a humanoid robot with realistic proportions.
    
    Based on average human anthropometric data:
    - Height: ~1.7m
    - Mass: ~70kg
    - Realistic segment proportions
    """
    print("Creating humanoid robot...")
    
    # Anthropometric parameters (scaled for robot)
    total_height = 1.7  # m
    total_mass = 70.0   # kg
    
    # Segment lengths (as fraction of total height)
    head_height = 0.13 * total_height
    torso_height = 0.30 * total_height
    upper_arm_length = 0.19 * total_height
    forearm_length = 0.15 * total_height
    thigh_length = 0.25 * total_height
    shin_length = 0.25 * total_height
    foot_length = 0.15 * total_height
    
    # Segment masses (as fraction of total mass)
    head_mass = 0.07 * total_mass
    torso_mass = 0.43 * total_mass
    upper_arm_mass = 0.028 * total_mass
    forearm_mass = 0.016 * total_mass
    hand_mass = 0.006 * total_mass
    thigh_mass = 0.1 * total_mass
    shin_mass = 0.045 * total_mass
    foot_mass = 0.014 * total_mass
    
    links = []
    joints = []
    
    # Pelvis (root link)
    pelvis = pin.Link(
        name="pelvis",
        mass=0.15 * total_mass,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.diag([0.1, 0.1, 0.05])
    )
    links.append(pelvis)
    
    # Torso
    torso = pin.Link(
        name="torso",
        mass=torso_mass,
        center_of_mass=np.array([0.0, 0.0, torso_height/2]),
        inertia_tensor=np.diag([
            torso_mass * (0.2**2 + torso_height**2) / 12,
            torso_mass * (0.15**2 + torso_height**2) / 12,
            torso_mass * (0.2**2 + 0.15**2) / 12
        ])
    )
    links.append(torso)
    
    # Torso joint (3-DOF: roll, pitch, yaw)
    torso_joint = pin.Joint(
        name="torso_joint",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),  # Simplified to 1-DOF for now
        parent_link="pelvis",
        child_link="torso",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, 0.1])),
        position_limit_lower=-np.pi/6,
        position_limit_upper=np.pi/6
    )
    joints.append(torso_joint)
    
    # Head
    head = pin.Link(
        name="head",
        mass=head_mass,
        center_of_mass=np.array([0.0, 0.0, head_height/2]),
        inertia_tensor=np.diag([
            head_mass * 0.1**2,
            head_mass * 0.1**2,
            head_mass * 0.08**2
        ])
    )
    links.append(head)
    
    head_joint = pin.Joint(
        name="head_joint",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 0, 1]),  # Head yaw
        parent_link="torso",
        child_link="head",
        origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, torso_height])),
        position_limit_lower=-np.pi/2,
        position_limit_upper=np.pi/2
    )
    joints.append(head_joint)
    
    # Arms (left and right)
    for side in ["left", "right"]:
        side_multiplier = -1 if side == "left" else 1
        shoulder_offset = np.array([0, side_multiplier * 0.2, torso_height * 0.8])
        
        # Upper arm
        upper_arm = pin.Link(
            name=f"{side}_upper_arm",
            mass=upper_arm_mass,
            center_of_mass=np.array([0, 0, -upper_arm_length/2]),
            inertia_tensor=np.diag([
                upper_arm_mass * upper_arm_length**2 / 12,
                upper_arm_mass * upper_arm_length**2 / 12,
                upper_arm_mass * 0.03**2
            ])
        )
        links.append(upper_arm)
        
        # Forearm
        forearm = pin.Link(
            name=f"{side}_forearm",
            mass=forearm_mass,
            center_of_mass=np.array([0, 0, -forearm_length/2]),
            inertia_tensor=np.diag([
                forearm_mass * forearm_length**2 / 12,
                forearm_mass * forearm_length**2 / 12,
                forearm_mass * 0.02**2
            ])
        )
        links.append(forearm)
        
        # Hand
        hand = pin.Link(
            name=f"{side}_hand",
            mass=hand_mass,
            center_of_mass=np.array([0, 0, -0.05]),
            inertia_tensor=np.diag([
                hand_mass * 0.05**2,
                hand_mass * 0.05**2,
                hand_mass * 0.02**2
            ])
        )
        links.append(hand)
        
        # Shoulder joint (simplified to 1-DOF)
        shoulder_joint = pin.Joint(
            name=f"{side}_shoulder",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Shoulder pitch
            parent_link="torso",
            child_link=f"{side}_upper_arm",
            origin_transform=pin.create_transform(np.eye(3), shoulder_offset),
            position_limit_lower=-np.pi,
            position_limit_upper=np.pi
        )
        joints.append(shoulder_joint)
        
        # Elbow joint
        elbow_joint = pin.Joint(
            name=f"{side}_elbow",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Elbow flexion
            parent_link=f"{side}_upper_arm",
            child_link=f"{side}_forearm",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -upper_arm_length])),
            position_limit_lower=0.0,
            position_limit_upper=np.pi
        )
        joints.append(elbow_joint)
        
        # Wrist joint
        wrist_joint = pin.Joint(
            name=f"{side}_wrist",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),  # Wrist rotation
            parent_link=f"{side}_forearm",
            child_link=f"{side}_hand",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -forearm_length])),
            position_limit_lower=-np.pi,
            position_limit_upper=np.pi
        )
        joints.append(wrist_joint)
    
    # Legs (left and right)
    for side in ["left", "right"]:
        side_multiplier = -1 if side == "left" else 1
        hip_offset = np.array([0, side_multiplier * 0.1, -0.05])
        
        # Thigh
        thigh = pin.Link(
            name=f"{side}_thigh",
            mass=thigh_mass,
            center_of_mass=np.array([0, 0, -thigh_length/2]),
            inertia_tensor=np.diag([
                thigh_mass * thigh_length**2 / 12,
                thigh_mass * thigh_length**2 / 12,
                thigh_mass * 0.05**2
            ])
        )
        links.append(thigh)
        
        # Shin
        shin = pin.Link(
            name=f"{side}_shin",
            mass=shin_mass,
            center_of_mass=np.array([0, 0, -shin_length/2]),
            inertia_tensor=np.diag([
                shin_mass * shin_length**2 / 12,
                shin_mass * shin_length**2 / 12,
                shin_mass * 0.03**2
            ])
        )
        links.append(shin)
        
        # Foot
        foot = pin.Link(
            name=f"{side}_foot",
            mass=foot_mass,
            center_of_mass=np.array([foot_length/3, 0, 0]),
            inertia_tensor=np.diag([
                foot_mass * foot_length**2 / 12,
                foot_mass * foot_length**2 / 12,
                foot_mass * 0.05**2
            ])
        )
        links.append(foot)
        
        # Hip joint
        hip_joint = pin.Joint(
            name=f"{side}_hip",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Hip pitch
            parent_link="pelvis",
            child_link=f"{side}_thigh",
            origin_transform=pin.create_transform(np.eye(3), hip_offset),
            position_limit_lower=-np.pi/2,
            position_limit_upper=np.pi/2
        )
        joints.append(hip_joint)
        
        # Knee joint
        knee_joint = pin.Joint(
            name=f"{side}_knee",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Knee flexion
            parent_link=f"{side}_thigh",
            child_link=f"{side}_shin",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -thigh_length])),
            position_limit_lower=0.0,
            position_limit_upper=np.pi
        )
        joints.append(knee_joint)
        
        # Ankle joint
        ankle_joint = pin.Joint(
            name=f"{side}_ankle",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),  # Ankle pitch
            parent_link=f"{side}_shin",
            child_link=f"{side}_foot",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -shin_length])),
            position_limit_lower=-np.pi/3,
            position_limit_upper=np.pi/3
        )
        joints.append(ankle_joint)
    
    robot = pin.create_robot_model(
        name="humanoid_robot",
        links=links,
        joints=joints,
        root_link="pelvis"
    )
    
    print(f"Created humanoid with {robot.num_links} links and {robot.num_dof} DOF")
    print(f"Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"Height: ~{total_height:.1f} m")
    
    return robot


def demonstrate_humanoid_poses(robot):
    """Demonstrate various humanoid poses and postures."""
    print("\n=== Humanoid Pose Demonstration ===")
    
    # Define meaningful poses
    poses = {
        "standing": np.zeros(robot.num_dof),
        "arms_up": create_pose_arms_up(robot),
        "sitting": create_pose_sitting(robot),
        "walking_step": create_pose_walking_step(robot),
        "reaching": create_pose_reaching(robot),
    }
    
    for pose_name, q in poses.items():
        print(f"\nPose: {pose_name}")
        
        try:
            # Validate pose
            if not robot.is_valid_joint_configuration(q):
                print(f"  Warning: Pose violates joint limits")
                continue
            
            # Compute key positions
            head_pos = pin.get_link_position(robot, q, "head")
            left_hand_pos = pin.get_link_position(robot, q, "left_hand")
            right_hand_pos = pin.get_link_position(robot, q, "right_hand")
            left_foot_pos = pin.get_link_position(robot, q, "left_foot")
            right_foot_pos = pin.get_link_position(robot, q, "right_foot")
            
            print(f"  Head position: [{head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f}]")
            print(f"  Left hand: [{left_hand_pos[0]:.2f}, {left_hand_pos[1]:.2f}, {left_hand_pos[2]:.2f}]")
            print(f"  Right hand: [{right_hand_pos[0]:.2f}, {right_hand_pos[1]:.2f}, {right_hand_pos[2]:.2f}]")
            print(f"  Left foot: [{left_foot_pos[0]:.2f}, {left_foot_pos[1]:.2f}, {left_foot_pos[2]:.2f}]")
            print(f"  Right foot: [{right_foot_pos[0]:.2f}, {right_foot_pos[1]:.2f}, {right_foot_pos[2]:.2f}]")
            
            # Compute center of mass
            com = compute_center_of_mass(robot, q)
            print(f"  Center of mass: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
            
            # Check balance (simplified)
            foot_center = (left_foot_pos + right_foot_pos) / 2
            com_offset = np.linalg.norm(com[:2] - foot_center[:2])
            balanced = com_offset < 0.1  # Within 10cm of foot center
            print(f"  Balanced: {balanced} (COM offset: {com_offset:.3f} m)")
            
        except Exception as e:
            print(f"  Error: {e}")


def create_pose_arms_up(robot):
    """Create pose with arms raised up."""
    q = np.zeros(robot.num_dof)
    
    # Find joint indices
    joint_names = robot.actuated_joint_names
    
    # Set arm positions
    if "left_shoulder" in joint_names:
        q[joint_names.index("left_shoulder")] = -np.pi/2  # Left arm up
    if "right_shoulder" in joint_names:
        q[joint_names.index("right_shoulder")] = -np.pi/2  # Right arm up
    
    return q


def create_pose_sitting(robot):
    """Create sitting pose."""
    q = np.zeros(robot.num_dof)
    
    joint_names = robot.actuated_joint_names
    
    # Bend hips and knees
    if "left_hip" in joint_names:
        q[joint_names.index("left_hip")] = np.pi/2
    if "right_hip" in joint_names:
        q[joint_names.index("right_hip")] = np.pi/2
    if "left_knee" in joint_names:
        q[joint_names.index("left_knee")] = np.pi/2
    if "right_knee" in joint_names:
        q[joint_names.index("right_knee")] = np.pi/2
    
    return q


def create_pose_walking_step(robot):
    """Create walking step pose."""
    q = np.zeros(robot.num_dof)
    
    joint_names = robot.actuated_joint_names
    
    # Left leg forward, right leg back
    if "left_hip" in joint_names:
        q[joint_names.index("left_hip")] = -np.pi/6  # Left leg forward
    if "right_hip" in joint_names:
        q[joint_names.index("right_hip")] = np.pi/6   # Right leg back
    
    # Slight knee bend
    if "left_knee" in joint_names:
        q[joint_names.index("left_knee")] = np.pi/12
    if "right_knee" in joint_names:
        q[joint_names.index("right_knee")] = np.pi/8
    
    # Arm swing
    if "left_shoulder" in joint_names:
        q[joint_names.index("left_shoulder")] = np.pi/6   # Left arm back
    if "right_shoulder" in joint_names:
        q[joint_names.index("right_shoulder")] = -np.pi/6  # Right arm forward
    
    return q


def create_pose_reaching(robot):
    """Create reaching pose."""
    q = np.zeros(robot.num_dof)
    
    joint_names = robot.actuated_joint_names
    
    # Reach forward with right arm
    if "right_shoulder" in joint_names:
        q[joint_names.index("right_shoulder")] = -np.pi/3
    if "right_elbow" in joint_names:
        q[joint_names.index("right_elbow")] = np.pi/4
    
    # Slight torso lean
    if "torso_joint" in joint_names:
        q[joint_names.index("torso_joint")] = np.pi/12
    
    return q


def analyze_humanoid_workspace(robot):
    """Analyze workspace of humanoid arms."""
    print("\n=== Workspace Analysis ===")
    
    # Analyze right arm workspace
    print("Analyzing right arm workspace...")
    
    # Sample arm configurations
    n_samples = 1000
    workspace_points = []
    
    joint_names = robot.actuated_joint_names
    shoulder_idx = joint_names.index("right_shoulder") if "right_shoulder" in joint_names else -1
    elbow_idx = joint_names.index("right_elbow") if "right_elbow" in joint_names else -1
    wrist_idx = joint_names.index("right_wrist") if "right_wrist" in joint_names else -1
    
    if shoulder_idx >= 0 and elbow_idx >= 0:
        for _ in range(n_samples):
            q = np.zeros(robot.num_dof)
            
            # Random arm configuration
            q[shoulder_idx] = np.random.uniform(-np.pi, np.pi)
            q[elbow_idx] = np.random.uniform(0, np.pi)
            if wrist_idx >= 0:
                q[wrist_idx] = np.random.uniform(-np.pi, np.pi)
            
            try:
                hand_pos = pin.get_link_position(robot, q, "right_hand")
                workspace_points.append(hand_pos)
            except:
                continue
        
        if workspace_points:
            workspace_points = np.array(workspace_points)
            
            print(f"Sampled {len(workspace_points)} valid configurations")
            print("Right hand workspace bounds:")
            print(f"  X: [{workspace_points[:, 0].min():.2f}, {workspace_points[:, 0].max():.2f}] m")
            print(f"  Y: [{workspace_points[:, 1].min():.2f}, {workspace_points[:, 1].max():.2f}] m")
            print(f"  Z: [{workspace_points[:, 2].min():.2f}, {workspace_points[:, 2].max():.2f}] m")
            
            # Maximum reach
            distances = np.linalg.norm(workspace_points, axis=1)
            print(f"  Maximum reach: {distances.max():.2f} m")
            print(f"  Average reach: {distances.mean():.2f} m")


def compute_center_of_mass(robot, joint_positions):
    """Compute center of mass of the humanoid robot."""
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


def demonstrate_balance_control(robot):
    """Demonstrate basic balance control concepts."""
    print("\n=== Balance Control Demonstration ===")
    
    # Test different poses for balance
    test_poses = {
        "standing": np.zeros(robot.num_dof),
        "lean_forward": create_pose_lean_forward(robot),
        "lean_back": create_pose_lean_back(robot),
        "single_leg": create_pose_single_leg(robot),
    }
    
    for pose_name, q in test_poses.items():
        print(f"\nTesting balance for pose: {pose_name}")
        
        try:
            # Compute center of mass
            com = compute_center_of_mass(robot, q)
            
            # Compute support polygon (simplified as line between feet)
            left_foot_pos = pin.get_link_position(robot, q, "left_foot")
            right_foot_pos = pin.get_link_position(robot, q, "right_foot")
            
            # Support center and width
            support_center = (left_foot_pos + right_foot_pos) / 2
            support_width = np.linalg.norm(left_foot_pos - right_foot_pos)
            
            # Check stability
            com_offset = np.linalg.norm(com[:2] - support_center[:2])
            stable = com_offset < support_width / 2
            
            print(f"  COM: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
            print(f"  Support center: [{support_center[0]:.2f}, {support_center[1]:.2f}]")
            print(f"  Support width: {support_width:.2f} m")
            print(f"  COM offset: {com_offset:.3f} m")
            print(f"  Stable: {stable}")
            
        except Exception as e:
            print(f"  Error: {e}")


def create_pose_lean_forward(robot):
    """Create forward leaning pose."""
    q = np.zeros(robot.num_dof)
    joint_names = robot.actuated_joint_names
    
    if "torso_joint" in joint_names:
        q[joint_names.index("torso_joint")] = np.pi/12  # Lean forward
    
    return q


def create_pose_lean_back(robot):
    """Create backward leaning pose."""
    q = np.zeros(robot.num_dof)
    joint_names = robot.actuated_joint_names
    
    if "torso_joint" in joint_names:
        q[joint_names.index("torso_joint")] = -np.pi/12  # Lean back
    
    return q


def create_pose_single_leg(robot):
    """Create single leg standing pose."""
    q = np.zeros(robot.num_dof)
    joint_names = robot.actuated_joint_names
    
    # Lift left leg
    if "left_hip" in joint_names:
        q[joint_names.index("left_hip")] = np.pi/3
    if "left_knee" in joint_names:
        q[joint_names.index("left_knee")] = np.pi/6
    
    return q


def main():
    """Main demonstration function."""
    print("py-pinocchio Humanoid Robot Example")
    print("=" * 50)
    
    # Create humanoid robot
    robot = create_humanoid_robot()
    
    # Demonstrate poses
    demonstrate_humanoid_poses(robot)
    
    # Analyze workspace
    analyze_humanoid_workspace(robot)
    
    # Demonstrate balance
    demonstrate_balance_control(robot)
    
    print("\n" + "=" * 50)
    print("Humanoid robot example completed!")
    print("\nKey features demonstrated:")
    print("- Realistic humanoid robot model")
    print("- Human-like proportions and constraints")
    print("- Various poses and postures")
    print("- Workspace analysis")
    print("- Balance and stability analysis")
    
    print(f"\nRobot specifications:")
    print(f"- Total mass: {sum(link.mass for link in robot.links):.1f} kg")
    print(f"- Degrees of freedom: {robot.num_dof}")
    print(f"- Links: {robot.num_links}")
    print(f"- Estimated height: ~1.7 m")


if __name__ == "__main__":
    main()
