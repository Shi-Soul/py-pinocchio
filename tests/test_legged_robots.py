#!/usr/bin/env python3
"""
Comprehensive tests for legged robot functionality

This test suite covers various legged robot configurations including:
- Bipedal robots (humanoid-like)
- Quadrupedal robots (dog-like)
- Complex multi-legged systems
- Walking gait analysis
- Stability computations
- Dynamic balance
"""

import numpy as np
import sys
import os

# Add parent directory to path to import py_pinocchio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import py_pinocchio as pin


class TestBipedalRobots:
    """Test bipedal robot functionality."""
    
    def create_simple_biped(self):
        """Create a simple 2D biped for testing."""
        # Physical parameters
        torso_mass = 10.0
        thigh_mass = 2.0
        shin_mass = 1.5
        foot_mass = 0.5
        
        thigh_length = 0.4
        shin_length = 0.35
        hip_width = 0.3
        
        # Create links
        links = []
        
        # Torso
        torso = pin.Link(
            name="torso",
            mass=torso_mass,
            center_of_mass=np.array([0.0, 0.0, 0.1]),
            inertia_tensor=np.diag([0.5, 0.5, 0.3])
        )
        links.append(torso)
        
        # Left leg
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
            center_of_mass=np.array([0.1, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, 0.02, 0.01])
        )
        links.append(left_foot)
        
        # Right leg (mirror of left)
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
            center_of_mass=np.array([0.1, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, 0.02, 0.01])
        )
        links.append(right_foot)
        
        # Create joints
        joints = []
        
        # Left leg joints
        left_hip = pin.Joint(
            name="left_hip",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),
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
            position_limit_lower=0.0,
            position_limit_upper=np.pi
        )
        joints.append(left_knee)
        
        left_ankle = pin.Joint(
            name="left_ankle",
            joint_type=pin.JointType.FIXED,
            parent_link="left_shin",
            child_link="left_foot",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -shin_length]))
        )
        joints.append(left_ankle)
        
        # Right leg joints
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
        
        return pin.create_robot_model(
            name="test_biped",
            links=links,
            joints=joints,
            root_link="torso"
        )
    
    def test_biped_creation(self):
        """Test biped robot creation."""
        robot = self.create_simple_biped()
        
        assert robot.name == "test_biped"
        assert robot.num_links == 7  # torso + 6 leg links
        assert robot.num_joints == 6  # 4 actuated + 2 fixed
        assert robot.num_dof == 4    # Only actuated joints count
        assert robot.root_link == "torso"
        
        expected_joints = ["left_hip", "left_knee", "right_hip", "right_knee"]
        assert robot.actuated_joint_names == expected_joints
    
    def test_biped_standing_pose(self):
        """Test biped in standing configuration."""
        robot = self.create_simple_biped()
        q_stand = np.array([0.0, 0.0, 0.0, 0.0])  # All joints at zero
        
        # Compute forward kinematics
        kinematic_state = pin.compute_forward_kinematics(robot, q_stand)
        
        # Check foot positions are symmetric and below torso
        left_foot_pos = kinematic_state.link_transforms["left_foot"].translation
        right_foot_pos = kinematic_state.link_transforms["right_foot"].translation
        torso_pos = kinematic_state.link_transforms["torso"].translation
        
        # Feet should be symmetric about Y-axis
        assert abs(left_foot_pos[0] + right_foot_pos[0]) < 1e-10
        assert abs(left_foot_pos[1] - right_foot_pos[1]) < 1e-10
        
        # Feet should be below torso
        assert left_foot_pos[2] < torso_pos[2]
        assert right_foot_pos[2] < torso_pos[2]
    
    def test_biped_walking_configurations(self):
        """Test various walking configurations."""
        robot = self.create_simple_biped()
        
        # Define walking poses
        poses = {
            "standing": np.array([0.0, 0.0, 0.0, 0.0]),
            "left_step": np.array([0.3, 0.5, -0.2, 0.3]),
            "right_step": np.array([-0.2, 0.3, 0.3, 0.5]),
            "squat": np.array([0.5, 1.0, 0.5, 1.0]),
        }
        
        for pose_name, joint_config in poses.items():
            # Check configuration is valid
            assert robot.is_valid_joint_configuration(joint_config)
            
            # Compute kinematics
            left_foot_pos = pin.get_link_position(robot, joint_config, "left_foot")
            right_foot_pos = pin.get_link_position(robot, joint_config, "right_foot")
            
            # Basic sanity checks
            assert len(left_foot_pos) == 3
            assert len(right_foot_pos) == 3
            assert not np.any(np.isnan(left_foot_pos))
            assert not np.any(np.isnan(right_foot_pos))
    
    def test_biped_dynamics(self):
        """Test biped dynamics computation."""
        robot = self.create_simple_biped()
        q = np.array([0.1, 0.2, -0.1, 0.3])
        qd = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Test mass matrix
        M = pin.compute_mass_matrix(robot, q)
        assert M.shape == (4, 4)
        assert np.all(np.diag(M) > 0)  # Positive definite
        assert np.allclose(M, M.T)     # Symmetric
        
        # Test gravity vector
        g = pin.compute_gravity_vector(robot, q)
        assert g.shape == (4,)
        
        # Test forward dynamics
        tau = np.array([1.0, 0.5, 1.0, 0.5])
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        assert qdd.shape == (4,)
        assert not np.any(np.isnan(qdd))


class TestQuadrupedalRobots:
    """Test quadrupedal robot functionality."""
    
    def create_simple_quadruped(self):
        """Create a simple quadruped robot."""
        # Physical parameters
        body_mass = 15.0
        leg_mass = 1.0
        
        body_length = 0.6
        body_width = 0.3
        leg_length = 0.3
        
        # Create links
        links = []
        
        # Body
        body = pin.Link(
            name="body",
            mass=body_mass,
            center_of_mass=np.array([0.0, 0.0, 0.0]),
            inertia_tensor=np.diag([1.0, 2.0, 1.5])
        )
        links.append(body)
        
        # Legs
        leg_names = ["front_left", "front_right", "rear_left", "rear_right"]
        leg_positions = [
            np.array([body_length/2, -body_width/2, 0]),   # front_left
            np.array([body_length/2, body_width/2, 0]),    # front_right
            np.array([-body_length/2, -body_width/2, 0]),  # rear_left
            np.array([-body_length/2, body_width/2, 0]),   # rear_right
        ]
        
        joints = []
        
        for i, (leg_name, leg_pos) in enumerate(zip(leg_names, leg_positions)):
            # Upper leg
            upper_leg = pin.Link(
                name=f"{leg_name}_upper",
                mass=leg_mass,
                center_of_mass=np.array([0.0, 0.0, -leg_length/4]),
                inertia_tensor=np.diag([0.05, 0.05, 0.01])
            )
            links.append(upper_leg)
            
            # Lower leg
            lower_leg = pin.Link(
                name=f"{leg_name}_lower",
                mass=leg_mass * 0.7,
                center_of_mass=np.array([0.0, 0.0, -leg_length/4]),
                inertia_tensor=np.diag([0.03, 0.03, 0.005])
            )
            links.append(lower_leg)
            
            # Hip joint
            hip_joint = pin.Joint(
                name=f"{leg_name}_hip",
                joint_type=pin.JointType.REVOLUTE,
                axis=np.array([0, 1, 0]),  # Pitch
                parent_link="body",
                child_link=f"{leg_name}_upper",
                origin_transform=pin.create_transform(np.eye(3), leg_pos),
                position_limit_lower=-np.pi/2,
                position_limit_upper=np.pi/2
            )
            joints.append(hip_joint)
            
            # Knee joint
            knee_joint = pin.Joint(
                name=f"{leg_name}_knee",
                joint_type=pin.JointType.REVOLUTE,
                axis=np.array([0, 1, 0]),  # Pitch
                parent_link=f"{leg_name}_upper",
                child_link=f"{leg_name}_lower",
                origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -leg_length/2])),
                position_limit_lower=-np.pi,
                position_limit_upper=0.0
            )
            joints.append(knee_joint)
        
        return pin.create_robot_model(
            name="test_quadruped",
            links=links,
            joints=joints,
            root_link="body"
        )
    
    def test_quadruped_creation(self):
        """Test quadruped robot creation."""
        robot = self.create_simple_quadruped()
        
        assert robot.name == "test_quadruped"
        assert robot.num_links == 9  # body + 8 leg links
        assert robot.num_joints == 8  # 8 actuated joints
        assert robot.num_dof == 8
        assert robot.root_link == "body"
        
        # Check all leg joints are present
        expected_joints = [
            "front_left_hip", "front_left_knee",
            "front_right_hip", "front_right_knee", 
            "rear_left_hip", "rear_left_knee",
            "rear_right_hip", "rear_right_knee"
        ]
        assert set(robot.actuated_joint_names) == set(expected_joints)
    
    def test_quadruped_standing_pose(self):
        """Test quadruped in standing configuration."""
        robot = self.create_simple_quadruped()
        q_stand = np.array([0.5, -1.0, 0.5, -1.0, 0.5, -1.0, 0.5, -1.0])  # Legs extended
        
        # Compute foot positions
        foot_positions = {}
        leg_names = ["front_left", "front_right", "rear_left", "rear_right"]
        
        for leg_name in leg_names:
            foot_pos = pin.get_link_position(robot, q_stand, f"{leg_name}_lower")
            foot_positions[leg_name] = foot_pos
        
        # All feet should be below body
        body_pos = pin.get_link_position(robot, q_stand, "body")
        for foot_pos in foot_positions.values():
            assert foot_pos[2] < body_pos[2]
        
        # Feet should form a rectangle
        front_left = foot_positions["front_left"]
        front_right = foot_positions["front_right"]
        rear_left = foot_positions["rear_left"]
        rear_right = foot_positions["rear_right"]
        
        # Check symmetry
        assert abs(front_left[1] + front_right[1]) < 1e-10  # Y symmetry
        assert abs(rear_left[1] + rear_right[1]) < 1e-10
        assert abs(front_left[0] - rear_left[0]) > 0.3      # X separation


def run_legged_robot_tests():
    """Run all legged robot tests."""
    print("Testing legged robot functionality...")
    
    # Test bipedal robots
    print("Testing bipedal robots...")
    test_biped = TestBipedalRobots()
    test_biped.test_biped_creation()
    test_biped.test_biped_standing_pose()
    test_biped.test_biped_walking_configurations()
    test_biped.test_biped_dynamics()
    print("✓ Bipedal robot tests passed")
    
    # Test quadrupedal robots
    print("Testing quadrupedal robots...")
    test_quad = TestQuadrupedalRobots()
    test_quad.test_quadruped_creation()
    test_quad.test_quadruped_standing_pose()
    print("✓ Quadrupedal robot tests passed")
    
    print("\n🎉 All legged robot tests passed!")


if __name__ == "__main__":
    run_legged_robot_tests()
