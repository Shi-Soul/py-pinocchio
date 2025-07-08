# Robot File Formats in py-pinocchio

This tutorial covers working with standard robot description formats including URDF (Universal Robot Description Format) and MJCF (MuJoCo XML Format). You'll learn how to load, parse, and convert between different formats.

## Table of Contents

1. [URDF Format](#urdf-format)
2. [MJCF Format](#mjcf-format)
3. [Loading and Parsing](#loading-and-parsing)
4. [Format Conversion](#format-conversion)
5. [Custom Extensions](#custom-extensions)
6. [Best Practices](#best-practices)

## URDF Format

URDF (Universal Robot Description Format) is the standard format for describing robots in ROS (Robot Operating System). It uses XML to define robot structure, kinematics, and dynamics.

### URDF Structure

A URDF file consists of:
- **Links**: Rigid bodies with inertial, visual, and collision properties
- **Joints**: Connections between links with kinematic constraints
- **Materials**: Visual appearance properties
- **Sensors**: Optional sensor definitions

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  
  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" 
               iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Link 1 -->
  <link name="link1">
    <inertial>
      <origin xyz="0.2 0 0" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" 
               iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
    
    <visual>
      <origin xyz="0.2 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    
    <collision>
      <origin xyz="0.2 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.4"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Joint 1 -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" 
           effort="100" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>
  
  <!-- Link 2 -->
  <link name="link2">
    <inertial>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" 
               iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0.15 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0.15 0 0" rpy="0 1.5708 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.3"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Joint 2 -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.4 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" 
           effort="50" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>
  
</robot>
```

### URDF Loading in py-pinocchio

```python
import py_pinocchio as pin
import numpy as np

def load_urdf_robot(urdf_path, verbose=True):
    """
    Load robot from URDF file.
    
    Args:
        urdf_path: Path to URDF file
        verbose: Print loading information
    
    Returns:
        Robot model
    """
    try:
        robot = pin.load_urdf(urdf_path)
        
        if verbose:
            print(f"Successfully loaded robot from {urdf_path}")
            print(f"Robot name: {robot.name}")
            print(f"Number of links: {robot.num_links}")
            print(f"Number of joints: {robot.num_joints}")
            print(f"Degrees of freedom: {robot.num_dof}")
            print(f"Actuated joints: {robot.actuated_joint_names}")
            
            # Print link information
            print("\nLink Information:")
            for link in robot.links:
                if link.has_inertia:
                    print(f"  {link.name}: mass={link.mass:.3f} kg")
                else:
                    print(f"  {link.name}: no inertia")
            
            # Print joint information
            print("\nJoint Information:")
            for joint in robot.joints:
                if joint.is_actuated:
                    print(f"  {joint.name}: {joint.joint_type.value}, "
                          f"limits=[{joint.position_limit_lower:.2f}, {joint.position_limit_upper:.2f}]")
        
        return robot
        
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return None

# Example usage
def create_example_urdf():
    """Create an example URDF file."""
    urdf_content = '''<?xml version="1.0"?>
<robot name="example_robot">
  
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
  </link>
  
  <link name="link1">
    <inertial>
      <origin xyz="0.25 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.25 0 0"/>
      <geometry>
        <cylinder radius="0.02" length="0.5"/>
      </geometry>
      <material name="blue"/>
    </visual>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1.0"/>
  </joint>
  
</robot>'''
    
    with open('example_robot.urdf', 'w') as f:
        f.write(urdf_content)
    
    return 'example_robot.urdf'

# Create and load example
urdf_file = create_example_urdf()
robot = load_urdf_robot(urdf_file)

if robot:
    # Test the loaded robot
    q = np.array([np.pi/4])
    ee_pos = pin.get_link_position(robot, q, "link1")
    print(f"\nEnd-effector position at 45°: {ee_pos}")
```

### Advanced URDF Features

```python
def create_complex_urdf():
    """Create URDF with advanced features."""
    urdf_content = '''<?xml version="1.0"?>
<robot name="complex_robot">
  
  <!-- Macro for creating similar links -->
  <xacro:macro name="arm_link" params="name mass length radius">
    <link name="${name}">
      <inertial>
        <origin xyz="${length/2} 0 0"/>
        <mass value="${mass}"/>
        <inertia ixx="${mass*radius*radius/2}" 
                 iyy="${mass*(3*radius*radius + length*length)/12}"
                 izz="${mass*(3*radius*radius + length*length)/12}"
                 ixy="0" ixz="0" iyz="0"/>
      </inertial>
      
      <visual>
        <origin xyz="${length/2} 0 0" rpy="0 1.5708 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
        <material name="steel">
          <color rgba="0.7 0.7 0.7 1.0"/>
        </material>
      </visual>
      
      <collision>
        <origin xyz="${length/2} 0 0" rpy="0 1.5708 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
      <material name="base_color">
        <color rgba="0.2 0.2 0.2 1.0"/>
      </material>
    </visual>
  </link>
  
  <!-- Arm links using macro -->
  <xacro:arm_link name="link1" mass="2.0" length="0.4" radius="0.03"/>
  <xacro:arm_link name="link2" mass="1.5" length="0.3" radius="0.025"/>
  <xacro:arm_link name="link3" mass="1.0" length="0.2" radius="0.02"/>
  
  <!-- Joints with different types -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>
  
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.4 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="50" velocity="1.5"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>
  
  <joint name="joint3" type="prismatic">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0.3 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.2" effort="20" velocity="0.5"/>
    <dynamics damping="0.1" friction="0.02"/>
  </joint>
  
  <!-- Sensors -->
  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
      <material name="camera_color">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
  
  <joint name="camera_joint" type="fixed">
    <parent link="link3"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05"/>
  </joint>
  
</robot>'''
    
    with open('complex_robot.urdf', 'w') as f:
        f.write(urdf_content)
    
    return 'complex_robot.urdf'
```

## MJCF Format

MJCF (MuJoCo XML Format) is used by the MuJoCo physics simulator. It has a different structure and capabilities compared to URDF.

### MJCF Structure

MJCF organizes robots using:
- **Worldbody**: Root element containing all bodies
- **Bodies**: Rigid bodies with geometry and inertia
- **Joints**: Kinematic constraints between bodies
- **Actuators**: Motors and other actuators
- **Assets**: Meshes, textures, and materials

### Basic MJCF Example

```xml
<mujoco model="simple_arm">
  
  <compiler angle="radian" coordinate="local"/>
  
  <default>
    <joint damping="0.1" frictionloss="0.01"/>
    <geom rgba="0.7 0.7 0.7 1"/>
  </default>
  
  <asset>
    <material name="blue" rgba="0 0 1 1"/>
    <material name="red" rgba="1 0 0 1"/>
  </asset>
  
  <worldbody>
    
    <!-- Base -->
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.05 0.05" material="blue"/>
      <inertial pos="0 0 0.025" mass="2.0" 
                diaginertia="0.01 0.01 0.01"/>
      
      <!-- Link 1 -->
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" 
               range="-3.14159 3.14159"/>
        <geom type="cylinder" size="0.03 0.2" 
              pos="0.2 0 0" quat="0.7071 0 0.7071 0" material="red"/>
        <inertial pos="0.2 0 0" mass="1.5" 
                  diaginertia="0.02 0.02 0.005"/>
        
        <!-- Link 2 -->
        <body name="link2" pos="0.4 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" 
                 range="-3.14159 3.14159"/>
          <geom type="cylinder" size="0.025 0.15" 
                pos="0.15 0 0" quat="0.7071 0 0.7071 0" material="blue"/>
          <inertial pos="0.15 0 0" mass="1.0" 
                    diaginertia="0.01 0.01 0.003"/>
        </body>
        
      </body>
    </body>
    
  </worldbody>
  
  <actuator>
    <motor joint="joint1" gear="100"/>
    <motor joint="joint2" gear="50"/>
  </actuator>
  
</mujoco>
```

### MJCF Loading

```python
def load_mjcf_robot(mjcf_path, verbose=True):
    """
    Load robot from MJCF file.
    
    Args:
        mjcf_path: Path to MJCF file
        verbose: Print loading information
    
    Returns:
        Robot model
    """
    try:
        robot = pin.load_mjcf(mjcf_path)
        
        if verbose:
            print(f"Successfully loaded robot from {mjcf_path}")
            print(f"Robot name: {robot.name}")
            print(f"Number of bodies: {robot.num_links}")
            print(f"Number of joints: {robot.num_joints}")
            print(f"Degrees of freedom: {robot.num_dof}")
        
        return robot
        
    except Exception as e:
        print(f"Error loading MJCF: {e}")
        return None

def create_example_mjcf():
    """Create an example MJCF file."""
    mjcf_content = '''<mujoco model="example_robot">
  
  <compiler angle="radian"/>
  
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05" rgba="0 0 1 1"/>
      <inertial mass="1.0" diaginertia="0.01 0.01 0.01"/>
      
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.25" 
              pos="0.25 0 0" quat="0.7071 0 0.7071 0" rgba="1 0 0 1"/>
        <inertial pos="0.25 0 0" mass="0.5" diaginertia="0.01 0.01 0.005"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint="joint1"/>
  </actuator>
  
</mujoco>'''
    
    with open('example_robot.xml', 'w') as f:
        f.write(mjcf_content)
    
    return 'example_robot.xml'

# Create and load MJCF example
mjcf_file = create_example_mjcf()
robot_mjcf = load_mjcf_robot(mjcf_file)
```

## Format Conversion

Converting between URDF and MJCF formats allows you to use robots across different simulators and tools.

```python
def convert_urdf_to_mjcf(urdf_path, mjcf_path):
    """
    Convert URDF to MJCF format.
    
    Args:
        urdf_path: Input URDF file path
        mjcf_path: Output MJCF file path
    """
    try:
        # Load URDF
        robot = pin.load_urdf(urdf_path)
        
        # Convert to MJCF
        pin.save_mjcf(robot, mjcf_path)
        
        print(f"Successfully converted {urdf_path} to {mjcf_path}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def convert_mjcf_to_urdf(mjcf_path, urdf_path):
    """
    Convert MJCF to URDF format.
    
    Args:
        mjcf_path: Input MJCF file path
        urdf_path: Output URDF file path
    """
    try:
        # Load MJCF
        robot = pin.load_mjcf(mjcf_path)
        
        # Convert to URDF
        pin.save_urdf(robot, urdf_path)
        
        print(f"Successfully converted {mjcf_path} to {urdf_path}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

# Example conversions
urdf_file = create_example_urdf()
mjcf_file = create_example_mjcf()

# Convert URDF to MJCF
convert_urdf_to_mjcf(urdf_file, 'converted_robot.xml')

# Convert MJCF to URDF
convert_mjcf_to_urdf(mjcf_file, 'converted_robot.urdf')
```

## Best Practices

### File Organization

```python
def validate_robot_file(file_path, file_type='auto'):
    """
    Validate robot description file.
    
    Args:
        file_path: Path to robot file
        file_type: 'urdf', 'mjcf', or 'auto'
    
    Returns:
        Validation results
    """
    import os
    
    if not os.path.exists(file_path):
        return {"valid": False, "error": "File not found"}
    
    # Auto-detect file type
    if file_type == 'auto':
        if file_path.endswith('.urdf'):
            file_type = 'urdf'
        elif file_path.endswith('.xml'):
            file_type = 'mjcf'
        else:
            return {"valid": False, "error": "Unknown file type"}
    
    try:
        # Load robot
        if file_type == 'urdf':
            robot = pin.load_urdf(file_path)
        elif file_type == 'mjcf':
            robot = pin.load_mjcf(file_path)
        else:
            return {"valid": False, "error": "Unsupported file type"}
        
        # Validation checks
        issues = []
        
        # Check for minimum requirements
        if robot.num_links < 2:
            issues.append("Robot should have at least 2 links")
        
        if robot.num_dof == 0:
            issues.append("Robot has no degrees of freedom")
        
        # Check mass properties
        total_mass = sum(link.mass for link in robot.links if link.has_inertia)
        if total_mass <= 0:
            issues.append("Robot has no mass")
        
        # Check for unrealistic values
        for link in robot.links:
            if link.has_inertia:
                if link.mass > 1000:  # kg
                    issues.append(f"Link {link.name} has very large mass: {link.mass} kg")
                
                eigenvals = np.linalg.eigvals(link.inertia_tensor)
                if np.any(eigenvals <= 0):
                    issues.append(f"Link {link.name} has invalid inertia tensor")
        
        # Check joint limits
        for joint in robot.joints:
            if joint.is_actuated:
                if joint.position_limit_lower >= joint.position_limit_upper:
                    issues.append(f"Joint {joint.name} has invalid limits")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "robot_info": {
                "name": robot.name,
                "num_links": robot.num_links,
                "num_dof": robot.num_dof,
                "total_mass": total_mass
            }
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

# Validate example files
print("Validating URDF file:")
urdf_validation = validate_robot_file(urdf_file)
print(f"Valid: {urdf_validation['valid']}")
if not urdf_validation['valid']:
    print(f"Issues: {urdf_validation.get('issues', [])}")

print("\nValidating MJCF file:")
mjcf_validation = validate_robot_file(mjcf_file)
print(f"Valid: {mjcf_validation['valid']}")
if not mjcf_validation['valid']:
    print(f"Issues: {mjcf_validation.get('issues', [])}")
```

This comprehensive file formats tutorial covers the essential aspects of working with URDF and MJCF files in py-pinocchio, including loading, parsing, conversion, and validation. The examples demonstrate practical applications for robotics development and simulation.
