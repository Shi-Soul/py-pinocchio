#!/usr/bin/env python3
"""
MJCF parsing example for py-pinocchio

This example demonstrates how to load and work with MJCF (MuJoCo XML) files.
MJCF is the native format used by the MuJoCo physics simulator.

We'll create a sample MJCF string and parse it to show the functionality.
"""

import numpy as np
import py_pinocchio as pin


def create_sample_mjcf():
    """Create a sample MJCF string for demonstration."""
    mjcf_content = """<?xml version="1.0"?>
<mujoco model="simple_arm">
  
  <worldbody>
    
    <!-- Base body -->
    <body name="base" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.05"/>
      
      <!-- First link -->
      <body name="link1" pos="0 0 0.1">
        <inertial mass="0.5" pos="0.25 0 0" diaginertia="0.05 0.05 0.05"/>
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
        <geom type="cylinder" size="0.05 0.25"/>
        
        <!-- Second link -->
        <body name="link2" pos="0.5 0 0">
          <inertial mass="0.3" pos="0.15 0 0" diaginertia="0.03 0.03 0.03"/>
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom type="cylinder" size="0.03 0.15"/>
          
          <!-- End effector -->
          <body name="end_effector" pos="0.3 0 0">
            <inertial mass="0.1" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <geom type="sphere" size="0.02"/>
          </body>
          
        </body>
      </body>
    </body>
    
  </worldbody>
  
</mujoco>"""
    return mjcf_content


def demonstrate_mjcf_parsing():
    """Demonstrate MJCF parsing functionality."""
    print("=== MJCF Parsing Demo ===")
    
    # Create sample MJCF
    mjcf_content = create_sample_mjcf()
    print("Created sample MJCF content")
    
    # Parse MJCF string
    try:
        robot = pin.parse_mjcf_string(mjcf_content)
        print(f"Successfully parsed MJCF!")
        print(f"Robot name: {robot.name}")
        print(f"Number of links: {robot.num_links}")
        print(f"Number of joints: {robot.num_joints}")
        print(f"Degrees of freedom: {robot.num_dof}")
        print(f"Root link: {robot.root_link}")
        
        return robot
        
    except Exception as e:
        print(f"Error parsing MJCF: {e}")
        return None


def analyze_mjcf_robot_structure(robot):
    """Analyze the structure of the parsed MJCF robot."""
    print("\n=== MJCF Robot Structure Analysis ===")
    
    print("\nLinks:")
    for link in robot.links:
        print(f"  {link.name}:")
        print(f"    Mass: {link.mass:.3f} kg")
        print(f"    Center of mass: {link.center_of_mass}")
        print(f"    Has inertia: {link.has_inertia}")
    
    print("\nJoints:")
    for joint in robot.joints:
        print(f"  {joint.name}:")
        print(f"    Type: {joint.joint_type}")
        print(f"    Parent: {joint.parent_link} -> Child: {joint.child_link}")
        print(f"    Axis: {joint.axis}")
        if joint.joint_type != pin.JointType.FIXED:
            print(f"    Position limits: [{joint.position_limit_lower:.3f}, {joint.position_limit_upper:.3f}]")
    
    print(f"\nActuated joints (in order): {robot.actuated_joint_names}")


def demonstrate_mjcf_kinematics(robot):
    """Demonstrate kinematics with the MJCF robot."""
    print("\n=== Kinematics with MJCF Robot ===")
    
    # Test different configurations
    configurations = [
        np.array([0.0, 0.0]),           # Zero configuration
        np.array([np.pi/2, np.pi/4]),   # 90° and 45°
        np.array([-np.pi/3, np.pi/6]),  # -60° and 30°
        np.array([np.pi, -np.pi/2]),    # 180° and -90°
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
            ee_pos = pin.get_link_position(robot, q, "end_effector")
            print(f"  End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            
            # Get all link positions
            print("  All link positions:")
            kinematic_state = pin.compute_forward_kinematics(robot, q)
            for link_name, transform in kinematic_state.link_transforms.items():
                if link_name != "world":  # Skip world link
                    pos = transform.translation
                    print(f"    {link_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Compute Jacobian
            jacobian = pin.compute_geometric_jacobian(robot, q, "end_effector")
            print(f"  Jacobian condition number: {np.linalg.cond(jacobian):.2f}")
            
        except Exception as e:
            print(f"  Error in computation: {e}")


def demonstrate_mjcf_dynamics(robot):
    """Demonstrate dynamics with the MJCF robot."""
    print("\n=== Dynamics with MJCF Robot ===")
    
    # Test configuration
    q = np.array([np.pi/4, np.pi/6])  # 45° and 30°
    qd = np.array([0.1, -0.2])        # Joint velocities
    tau = np.array([1.0, 0.5])        # Applied torques
    
    print(f"Configuration:")
    print(f"  Positions: {np.degrees(q)} degrees")
    print(f"  Velocities: {qd} rad/s")
    print(f"  Torques: {tau} N⋅m")
    
    try:
        # Compute dynamics quantities
        mass_matrix = pin.compute_mass_matrix(robot, q)
        coriolis_forces = pin.compute_coriolis_forces(robot, q, qd)
        gravity_forces = pin.compute_gravity_vector(robot, q)
        
        print(f"\nDynamics Analysis:")
        print(f"  Mass matrix:")
        print(f"    {mass_matrix}")
        print(f"  Coriolis forces: {coriolis_forces}")
        print(f"  Gravity forces: {gravity_forces}")
        
        # Compute forward dynamics
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        print(f"  Resulting accelerations: {qdd} rad/s²")
        
        # Verify with inverse dynamics
        tau_computed = pin.compute_inverse_dynamics(robot, q, qd, qdd)
        print(f"  Inverse dynamics verification:")
        print(f"    Applied torques: {tau}")
        print(f"    Computed torques: {tau_computed}")
        print(f"    Difference: {np.abs(tau - tau_computed)}")
        
    except Exception as e:
        print(f"  Error in dynamics computation: {e}")


def compare_mjcf_vs_urdf():
    """Compare MJCF and URDF parsing for similar robots."""
    print("\n=== MJCF vs URDF Comparison ===")
    
    # Parse MJCF robot
    mjcf_content = create_sample_mjcf()
    mjcf_robot = pin.parse_mjcf_string(mjcf_content)
    
    # Create equivalent URDF content
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_arm">
  
  <link name="world"/>
  
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.25 0 0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <link name="link2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0.15 0 0"/>
      <inertia ixx="0.03" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base"/>
    <origin xyz="0 0 0"/>
  </joint>
  
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159"/>
  </joint>
  
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.5 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
  
</robot>"""
    
    urdf_robot = pin.parse_urdf_string(urdf_content)
    
    print("Comparison Results:")
    print(f"  MJCF Robot:")
    print(f"    Links: {mjcf_robot.num_links}, Joints: {mjcf_robot.num_joints}, DOF: {mjcf_robot.num_dof}")
    print(f"  URDF Robot:")
    print(f"    Links: {urdf_robot.num_links}, Joints: {urdf_robot.num_joints}, DOF: {urdf_robot.num_dof}")
    
    # Test same configuration on both
    q = np.array([0.5, -0.3])
    
    try:
        mjcf_ee_pos = pin.get_link_position(mjcf_robot, q, "end_effector")
        urdf_ee_pos = pin.get_link_position(urdf_robot, q, "link2")
        
        print(f"  End-effector positions at q={q}:")
        print(f"    MJCF: {mjcf_ee_pos}")
        print(f"    URDF: {urdf_ee_pos}")
        
    except Exception as e:
        print(f"  Error in comparison: {e}")


def save_sample_mjcf_file():
    """Save the sample MJCF to a file for reference."""
    mjcf_content = create_sample_mjcf()
    
    try:
        with open("examples/sample_robot.xml", "w") as f:
            f.write(mjcf_content)
        print("\nSaved sample MJCF to 'examples/sample_robot.xml'")
        print("You can now load it with: robot = pin.parse_mjcf_file('examples/sample_robot.xml')")
    except Exception as e:
        print(f"Could not save MJCF file: {e}")


def main():
    """Main demonstration function."""
    print("py-pinocchio MJCF Example")
    print("=" * 40)
    
    # Parse MJCF
    robot = demonstrate_mjcf_parsing()
    
    if robot is None:
        print("Failed to parse MJCF. Exiting.")
        return
    
    # Analyze robot structure
    analyze_mjcf_robot_structure(robot)
    
    # Demonstrate kinematics
    demonstrate_mjcf_kinematics(robot)
    
    # Demonstrate dynamics
    demonstrate_mjcf_dynamics(robot)
    
    # Compare with URDF
    compare_mjcf_vs_urdf()
    
    # Save sample file
    save_sample_mjcf_file()
    
    print("\n" + "=" * 40)
    print("MJCF example completed successfully!")
    print("\nTips for working with MJCF files:")
    print("- Use pin.parse_mjcf_file('path/to/robot.xml')")
    print("- MJCF uses different conventions than URDF (quaternions, etc.)")
    print("- Check robot.num_dof to see how many joints are actuated")
    print("- MJCF worldbody defines the kinematic tree structure")


if __name__ == "__main__":
    main()
