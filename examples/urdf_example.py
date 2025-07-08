#!/usr/bin/env python3
"""
URDF parsing example for py-pinocchio

This example demonstrates how to load and work with URDF files.
Since we don't have actual URDF files, we'll create a simple URDF
string and parse it to show the functionality.
"""

import numpy as np
import py_pinocchio as pin

def create_sample_urdf():
    """Create a sample URDF string for demonstration."""
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_arm">
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
    </visual>
  </link>
  
  <!-- First link -->
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.25 0 0" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </visual>
  </link>
  
  <!-- Second link -->
  <link name="link2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.3"/>
      </geometry>
    </visual>
  </link>
  
  <!-- Joint 1: base to link1 -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" velocity="2.0" effort="10.0"/>
    <dynamics damping="0.1" friction="0.05"/>
  </joint>
  
  <!-- Joint 2: link1 to link2 -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="2.0" effort="5.0"/>
    <dynamics damping="0.05" friction="0.02"/>
  </joint>
  
</robot>"""
    return urdf_content


def demonstrate_urdf_parsing():
    """Demonstrate URDF parsing functionality."""
    print("=== URDF Parsing Demo ===")
    
    # Create sample URDF
    urdf_content = create_sample_urdf()
    print("Created sample URDF content")
    
    # Parse URDF string
    try:
        robot = pin.parse_urdf_string(urdf_content)
        print(f"Successfully parsed URDF!")
        print(f"Robot name: {robot.name}")
        print(f"Number of links: {robot.num_links}")
        print(f"Number of joints: {robot.num_joints}")
        print(f"Degrees of freedom: {robot.num_dof}")
        print(f"Root link: {robot.root_link}")
        
        return robot
        
    except Exception as e:
        print(f"Error parsing URDF: {e}")
        return None


def analyze_robot_structure(robot):
    """Analyze the structure of the parsed robot."""
    print("\n=== Robot Structure Analysis ===")
    
    print("\nLinks:")
    for link in robot.links:
        print(f"  {link.name}:")
        print(f"    Mass: {link.mass:.3f} kg")
        print(f"    Center of mass: {link.center_of_mass}")
        print(f"    Has inertia: {link.has_inertia}")
        if link.visual_geometry:
            print(f"    Visual geometry: {link.visual_geometry}")
    
    print("\nJoints:")
    for joint in robot.joints:
        print(f"  {joint.name}:")
        print(f"    Type: {joint.joint_type}")
        print(f"    Parent: {joint.parent_link} -> Child: {joint.child_link}")
        print(f"    Axis: {joint.axis}")
        print(f"    Position limits: [{joint.position_limit_lower:.3f}, {joint.position_limit_upper:.3f}]")
        print(f"    Velocity limit: {joint.velocity_limit:.3f}")
        print(f"    Effort limit: {joint.effort_limit:.3f}")
        print(f"    Damping: {joint.damping:.3f}, Friction: {joint.friction:.3f}")
    
    print(f"\nActuated joints (in order): {robot.actuated_joint_names}")


def demonstrate_kinematics_with_urdf_robot(robot):
    """Demonstrate kinematics with the URDF robot."""
    print("\n=== Kinematics with URDF Robot ===")
    
    # Test different configurations
    configurations = [
        np.array([0.0, 0.0]),           # Zero configuration
        np.array([np.pi/2, np.pi/4]),   # 90° and 45°
        np.array([-np.pi/3, np.pi/6]),  # -60° and 30°
    ]
    
    for i, q in enumerate(configurations):
        print(f"\nConfiguration {i+1}: {np.degrees(q)} degrees")
        
        # Check if configuration is valid
        if not robot.is_valid_joint_configuration(q):
            print("  Configuration exceeds joint limits!")
            continue
        
        # Compute forward kinematics
        try:
            # Get end-effector position
            ee_pos = pin.get_link_position(robot, q, "link2")
            print(f"  End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            
            # Compute Jacobian
            jacobian = pin.compute_geometric_jacobian(robot, q, "link2")
            print(f"  Jacobian condition number: {np.linalg.cond(jacobian):.2f}")
            
            # Compute some dynamics quantities
            gravity_torques = pin.compute_gravity_vector(robot, q)
            print(f"  Gravity torques: [{gravity_torques[0]:.3f}, {gravity_torques[1]:.3f}] N⋅m")
            
        except Exception as e:
            print(f"  Error in computation: {e}")


def demonstrate_joint_limits_checking(robot):
    """Demonstrate joint limits checking."""
    print("\n=== Joint Limits Checking ===")
    
    test_configurations = [
        np.array([0.0, 0.0]),           # Valid
        np.array([np.pi, 0.0]),         # Valid (at limit)
        np.array([4.0, 0.0]),           # Invalid (exceeds limit)
        np.array([0.0, 2.0]),           # Invalid (exceeds limit)
        np.array([-4.0, -2.0]),         # Invalid (both exceed limits)
    ]
    
    for i, q in enumerate(test_configurations):
        is_valid = robot.is_valid_joint_configuration(q)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  Config {i+1}: {np.degrees(q)} deg -> {status}")


def save_sample_urdf_file():
    """Save the sample URDF to a file for reference."""
    urdf_content = create_sample_urdf()
    
    try:
        with open("examples/sample_robot.urdf", "w") as f:
            f.write(urdf_content)
        print("\nSaved sample URDF to 'examples/sample_robot.urdf'")
        print("You can now load it with: robot = pin.parse_urdf_file('examples/sample_robot.urdf')")
    except Exception as e:
        print(f"Could not save URDF file: {e}")


def main():
    """Main demonstration function."""
    print("py-pinocchio URDF Example")
    print("=" * 40)
    
    # Parse URDF
    robot = demonstrate_urdf_parsing()
    
    if robot is None:
        print("Failed to parse URDF. Exiting.")
        return
    
    # Analyze robot structure
    analyze_robot_structure(robot)
    
    # Demonstrate kinematics
    demonstrate_kinematics_with_urdf_robot(robot)
    
    # Demonstrate joint limits
    demonstrate_joint_limits_checking(robot)
    
    # Save sample file
    save_sample_urdf_file()
    
    print("\n" + "=" * 40)
    print("URDF example completed successfully!")
    print("\nTips for working with real URDF files:")
    print("- Use pin.parse_urdf_file('path/to/robot.urdf')")
    print("- Check robot.num_dof to see how many joints are actuated")
    print("- Use robot.actuated_joint_names to see joint order")
    print("- Always validate joint configurations with is_valid_joint_configuration()")


if __name__ == "__main__":
    main()
