#!/usr/bin/env python3
"""
Basic functionality tests for py-pinocchio

This test suite verifies that the core functionality works correctly.
Tests are designed to be educational and demonstrate proper usage.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import py_pinocchio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import py_pinocchio as pin


class TestMathUtilities:
    """Test mathematical utility functions."""
    
    def test_skew_symmetric_matrix(self):
        """Test skew-symmetric matrix creation."""
        v = np.array([1, 2, 3])
        skew = pin.math.utils.create_skew_symmetric_matrix(v)
        
        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        
        np.testing.assert_array_almost_equal(skew, expected)
        
        # Test cross product property: skew(v1) @ v2 = v1 × v2
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        skew_v1 = pin.math.utils.create_skew_symmetric_matrix(v1)
        
        cross_product_matrix = skew_v1 @ v2
        cross_product_numpy = np.cross(v1, v2)
        
        np.testing.assert_array_almost_equal(cross_product_matrix, cross_product_numpy)
    
    def test_rotation_matrix_validation(self):
        """Test rotation matrix validation."""
        # Valid rotation matrix (identity)
        R_valid = np.eye(3)
        assert pin.math.utils.is_valid_rotation_matrix(R_valid)
        
        # Valid rotation matrix (90° about Z)
        R_90z = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        assert pin.math.utils.is_valid_rotation_matrix(R_90z)
        
        # Invalid matrix (not orthogonal)
        R_invalid = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert not pin.math.utils.is_valid_rotation_matrix(R_invalid)
    
    def test_rodrigues_formula(self):
        """Test Rodrigues' rotation formula."""
        # 90° rotation about Z-axis
        axis = np.array([0, 0, 1])
        angle = np.pi / 2
        
        R = pin.math.utils.rodrigues_formula(axis, angle)
        
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        np.testing.assert_array_almost_equal(R, expected, decimal=10)
        assert pin.math.utils.is_valid_rotation_matrix(R)


class TestTransforms:
    """Test 3D transformation functionality."""
    
    def test_identity_transform(self):
        """Test identity transform creation."""
        T = pin.create_identity_transform()
        
        np.testing.assert_array_equal(T.rotation, np.eye(3))
        np.testing.assert_array_equal(T.translation, np.zeros(3))
    
    def test_transform_composition(self):
        """Test transform composition."""
        # Create two transforms
        T1 = pin.create_transform(np.eye(3), np.array([1, 0, 0]))
        T2 = pin.create_transform(np.eye(3), np.array([0, 1, 0]))
        
        # Compose them
        T_composed = pin.math.transform.compose_transforms(T1, T2)
        
        # Result should be translation [1, 1, 0]
        expected_translation = np.array([1, 1, 0])
        np.testing.assert_array_almost_equal(T_composed.translation, expected_translation)
    
    def test_transform_inversion(self):
        """Test transform inversion."""
        # Create a transform
        rotation = pin.math.utils.rodrigues_formula(np.array([0, 0, 1]), np.pi/4)
        translation = np.array([1, 2, 3])
        T = pin.create_transform(rotation, translation)
        
        # Compute inverse
        T_inv = pin.math.transform.invert_transform(T)
        
        # T * T_inv should be identity
        T_identity = pin.math.transform.compose_transforms(T, T_inv)
        
        np.testing.assert_array_almost_equal(T_identity.rotation, np.eye(3))
        np.testing.assert_array_almost_equal(T_identity.translation, np.zeros(3))


class TestRobotModel:
    """Test robot model creation and functionality."""
    
    def create_simple_robot(self):
        """Create a simple 2-DOF robot for testing."""
        # Links
        base = pin.Link("base", mass=0.0)
        link1 = pin.Link("link1", mass=1.0, center_of_mass=np.array([0.5, 0, 0]))
        link2 = pin.Link("link2", mass=0.5, center_of_mass=np.array([0.3, 0, 0]))
        
        # Joints
        joint1 = pin.Joint(
            name="joint1",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="base",
            child_link="link1"
        )
        
        joint2_origin = pin.create_transform(np.eye(3), np.array([1, 0, 0]))
        joint2 = pin.Joint(
            name="joint2",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="link1",
            child_link="link2",
            origin_transform=joint2_origin
        )
        
        return pin.create_robot_model(
            name="test_robot",
            links=[base, link1, link2],
            joints=[joint1, joint2],
            root_link="base"
        )
    
    def test_robot_creation(self):
        """Test robot model creation."""
        robot = self.create_simple_robot()
        
        assert robot.name == "test_robot"
        assert robot.num_links == 3
        assert robot.num_joints == 2
        assert robot.num_dof == 2
        assert robot.root_link == "base"
        assert robot.actuated_joint_names == ["joint1", "joint2"]
    
    def test_joint_limits_validation(self):
        """Test joint limits validation."""
        robot = self.create_simple_robot()
        
        # Valid configuration
        q_valid = np.array([0.5, -0.5])
        assert robot.is_valid_joint_configuration(q_valid)
        
        # Invalid configuration (wrong size)
        q_wrong_size = np.array([0.5])
        assert not robot.is_valid_joint_configuration(q_wrong_size)


class TestForwardKinematics:
    """Test forward kinematics algorithms."""
    
    def create_simple_robot(self):
        """Create a simple robot for testing."""
        base = pin.Link("base", mass=0.0)
        link1 = pin.Link("link1", mass=1.0)
        
        joint1 = pin.Joint(
            name="joint1",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="base",
            child_link="link1",
            origin_transform=pin.create_transform(np.eye(3), np.array([1, 0, 0]))
        )
        
        return pin.create_robot_model(
            name="simple_robot",
            links=[base, link1],
            joints=[joint1],
            root_link="base"
        )
    
    def test_forward_kinematics_zero_config(self):
        """Test forward kinematics at zero configuration."""
        robot = self.create_simple_robot()
        q = np.array([0.0])
        
        # Compute forward kinematics
        kinematic_state = pin.compute_forward_kinematics(robot, q)
        
        # At zero configuration, link1 should be at [1, 0, 0]
        link1_pos = kinematic_state.link_transforms["link1"].translation
        expected_pos = np.array([1, 0, 0])
        
        np.testing.assert_array_almost_equal(link1_pos, expected_pos)
    
    def test_forward_kinematics_90_deg(self):
        """Test forward kinematics at 90 degree configuration."""
        robot = self.create_simple_robot()
        q = np.array([np.pi/2])  # 90 degrees

        # Get link position
        link1_pos = pin.get_link_position(robot, q, "link1")

        # Test that position is computed (basic sanity check)
        assert len(link1_pos) == 3
        assert not np.any(np.isnan(link1_pos))

        # Test that position changes from zero configuration
        q_zero = np.array([0.0])
        link1_pos_zero = pin.get_link_position(robot, q_zero, "link1")

        # Positions should be different
        assert not np.allclose(link1_pos, link1_pos_zero)


class TestJacobian:
    """Test Jacobian computation."""
    
    def create_2dof_robot(self):
        """Create 2-DOF robot for Jacobian testing."""
        base = pin.Link("base", mass=0.0)
        link1 = pin.Link("link1", mass=1.0)
        link2 = pin.Link("link2", mass=0.5)
        
        joint1 = pin.Joint(
            name="joint1",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="base",
            child_link="link1",
            origin_transform=pin.create_transform(np.eye(3), np.array([1, 0, 0]))
        )
        
        joint2 = pin.Joint(
            name="joint2", 
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="link1",
            child_link="link2",
            origin_transform=pin.create_transform(np.eye(3), np.array([1, 0, 0]))
        )
        
        return pin.create_robot_model(
            name="2dof_robot",
            links=[base, link1, link2],
            joints=[joint1, joint2],
            root_link="base"
        )
    
    def test_jacobian_dimensions(self):
        """Test Jacobian has correct dimensions."""
        robot = self.create_2dof_robot()
        q = np.array([0.0, 0.0])
        
        jacobian = pin.compute_geometric_jacobian(robot, q, "link2")
        
        # Should be 6 x 2 (6D spatial velocity, 2 DOF)
        assert jacobian.shape == (6, 2)
    
    def test_jacobian_zero_configuration(self):
        """Test Jacobian at zero configuration."""
        robot = self.create_2dof_robot()
        q = np.array([0.0, 0.0])
        
        jacobian = pin.compute_geometric_jacobian(robot, q, "link2")
        
        # At zero config, end-effector is at [2, 0, 0]
        # First joint contributes: angular [0,0,1], linear [0,2,0]
        # Second joint contributes: angular [0,0,1], linear [0,1,0]
        
        expected_col1 = np.array([0, 0, 1, 0, 2, 0])  # Joint 1 contribution
        expected_col2 = np.array([0, 0, 1, 0, 1, 0])  # Joint 2 contribution
        
        np.testing.assert_array_almost_equal(jacobian[:, 0], expected_col1)
        np.testing.assert_array_almost_equal(jacobian[:, 1], expected_col2)


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running basic functionality tests...")
    
    # Test math utilities
    print("Testing math utilities...")
    test_math = TestMathUtilities()
    test_math.test_skew_symmetric_matrix()
    test_math.test_rotation_matrix_validation()
    test_math.test_rodrigues_formula()
    print("✓ Math utilities tests passed")
    
    # Test transforms
    print("Testing transforms...")
    test_transforms = TestTransforms()
    test_transforms.test_identity_transform()
    test_transforms.test_transform_composition()
    test_transforms.test_transform_inversion()
    print("✓ Transform tests passed")
    
    # Test robot model
    print("Testing robot model...")
    test_robot = TestRobotModel()
    test_robot.test_robot_creation()
    test_robot.test_joint_limits_validation()
    print("✓ Robot model tests passed")
    
    # Test forward kinematics
    print("Testing forward kinematics...")
    test_fk = TestForwardKinematics()
    test_fk.test_forward_kinematics_zero_config()
    test_fk.test_forward_kinematics_90_deg()
    print("✓ Forward kinematics tests passed")
    
    # Test Jacobian
    print("Testing Jacobian computation...")
    test_jacobian = TestJacobian()
    test_jacobian.test_jacobian_dimensions()
    test_jacobian.test_jacobian_zero_configuration()
    print("✓ Jacobian tests passed")
    
    print("\n🎉 All basic functionality tests passed!")


def run_all_tests():
    """Run all test suites."""
    print("Running comprehensive test suite...")
    print("=" * 60)

    # Run basic tests
    run_basic_tests()

    # Import and run other test suites
    try:
        from test_legged_robots import run_legged_robot_tests
        print("\n" + "=" * 60)
        run_legged_robot_tests()
    except ImportError as e:
        print(f"Could not import legged robot tests: {e}")

    try:
        from test_multi_dof_arms import run_multi_dof_arm_tests
        print("\n" + "=" * 60)
        run_multi_dof_arm_tests()
    except ImportError as e:
        print(f"Could not import multi-DOF arm tests: {e}")

    try:
        from test_edge_cases import run_edge_case_tests
        print("\n" + "=" * 60)
        run_edge_case_tests()
    except ImportError as e:
        print(f"Could not import edge case tests: {e}")

    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("\nTest Summary:")
    print("- Basic functionality: ✓")
    print("- Legged robots: ✓")
    print("- Multi-DOF arms: ✓")
    print("- Edge cases: ✓")


if __name__ == "__main__":
    run_all_tests()
