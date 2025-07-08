# Robot Modeling in py-pinocchio

This tutorial covers advanced robot modeling techniques in py-pinocchio. You'll learn how to create complex robot models with proper mass properties, joint constraints, and kinematic chains.

## Table of Contents

1. [Link Properties](#link-properties)
2. [Joint Types and Constraints](#joint-types-and-constraints)
3. [Kinematic Chains](#kinematic-chains)
4. [Mass Properties](#mass-properties)
5. [Complex Robot Examples](#complex-robot-examples)
6. [Validation and Debugging](#validation-and-debugging)

## Link Properties

### Basic Link Creation

A link represents a rigid body in the robot:

```python
import numpy as np
import py_pinocchio as pin

# Minimal link (mass only)
simple_link = pin.Link(
    name="simple_link",
    mass=2.5  # kg
)

# Complete link with all properties
complete_link = pin.Link(
    name="complete_link",
    mass=5.0,  # kg
    center_of_mass=np.array([0.1, 0.0, 0.05]),  # m
    inertia_tensor=np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0], 
        [0.0, 0.0, 0.15]
    ])  # kg⋅m²
)
```

### Center of Mass

The center of mass (COM) is crucial for accurate dynamics:

```python
# For a uniform rod of length L along X-axis
rod_length = 0.5  # m
rod_com = np.array([rod_length/2, 0.0, 0.0])

# For a uniform cylinder along Z-axis
cylinder_height = 0.3  # m
cylinder_com = np.array([0.0, 0.0, cylinder_height/2])

# For complex shapes, use CAD software or measurement
complex_com = np.array([0.123, -0.045, 0.067])
```

### Inertia Tensors

Inertia tensors describe how mass is distributed. For a rigid body, the inertia tensor is:

$$\mathbf{I} = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix}$$

Where the diagonal elements are moments of inertia and off-diagonal elements are products of inertia. For common geometric shapes:

**Solid Cylinder** (radius $r$, height $h$, mass $m$):
$$I_{xx} = I_{yy} = \frac{m(3r^2 + h^2)}{12}, \quad I_{zz} = \frac{mr^2}{2}$$

```python
def cylinder_inertia(mass, radius, height):
    Ixx = Iyy = mass * (3*radius**2 + height**2) / 12
    Izz = mass * radius**2 / 2
    return np.diag([Ixx, Iyy, Izz])
```

**Solid Box** (width $w$, depth $d$, height $h$, mass $m$):
$$I_{xx} = \frac{m(d^2 + h^2)}{12}, \quad I_{yy} = \frac{m(w^2 + h^2)}{12}, \quad I_{zz} = \frac{m(w^2 + d^2)}{12}$$

```python
def box_inertia(mass, width, depth, height):
    Ixx = mass * (depth**2 + height**2) / 12
    Iyy = mass * (width**2 + height**2) / 12
    Izz = mass * (width**2 + depth**2) / 12
    return np.diag([Ixx, Iyy, Izz])
```

**Solid Sphere** (radius $r$, mass $m$):
$$I_{xx} = I_{yy} = I_{zz} = \frac{2mr^2}{5}$$

```python
def sphere_inertia(mass, radius):
    I = 2 * mass * radius**2 / 5
    return np.diag([I, I, I])
```

# Example usage
link_mass = 3.0
link_radius = 0.05
link_height = 0.4

link = pin.Link(
    name="cylinder_link",
    mass=link_mass,
    center_of_mass=np.array([0, 0, link_height/2]),
    inertia_tensor=cylinder_inertia(link_mass, link_radius, link_height)
)
```

## Joint Types and Constraints

### Revolute Joints

Most common joint type for rotational motion:

```python
# Standard revolute joint
revolute_joint = pin.Joint(
    name="shoulder_pitch",
    joint_type=pin.JointType.REVOLUTE,
    axis=np.array([0, 1, 0]),  # Rotation about Y-axis
    parent_link="torso",
    child_link="upper_arm",
    origin_transform=pin.create_transform(
        rotation=np.eye(3),
        translation=np.array([0, 0.2, 0.5])  # Joint location
    ),
    position_limit_lower=-np.pi/2,  # -90 degrees
    position_limit_upper=np.pi/2    # +90 degrees
)
```

### Prismatic Joints

For linear motion:

```python
# Linear actuator
prismatic_joint = pin.Joint(
    name="linear_actuator",
    joint_type=pin.JointType.PRISMATIC,
    axis=np.array([0, 0, 1]),  # Translation along Z-axis
    parent_link="base",
    child_link="platform",
    position_limit_lower=0.0,   # Minimum extension
    position_limit_upper=0.5    # Maximum extension (0.5m)
)
```

### Fixed Joints

For rigid connections:

```python
# Sensor mount (no relative motion)
fixed_joint = pin.Joint(
    name="sensor_mount",
    joint_type=pin.JointType.FIXED,
    parent_link="head",
    child_link="camera",
    origin_transform=pin.create_transform(
        rotation=np.eye(3),
        translation=np.array([0.1, 0, 0.05])
    )
)
```

### Joint Limits

Always specify realistic joint limits:

```python
# Human-like joint limits
human_joints = {
    "shoulder_pitch": (-np.pi, np.pi/2),      # -180° to +90°
    "shoulder_roll": (-np.pi/2, np.pi/2),     # -90° to +90°
    "elbow_flexion": (0, np.pi),              # 0° to 180°
    "wrist_rotation": (-np.pi, np.pi),        # -180° to +180°
}

# Industrial robot limits (more restrictive)
industrial_joints = {
    "joint_1": (-np.pi, np.pi),               # Base rotation
    "joint_2": (-np.pi/2, np.pi/2),          # Shoulder
    "joint_3": (-np.pi, 0),                   # Elbow (negative only)
    "joint_4": (-np.pi, np.pi),               # Wrist roll
    "joint_5": (-np.pi/2, np.pi/2),          # Wrist pitch
    "joint_6": (-np.pi, np.pi),               # Wrist yaw
}
```

## Kinematic Chains

### Serial Chains

Most robot arms are serial chains:

```python
def create_serial_arm(num_links=6):
    """Create a serial manipulator with specified number of links."""
    links = []
    joints = []
    
    # Base link
    base = pin.Link(name="base", mass=10.0)
    links.append(base)
    
    # Create chain of links and joints
    for i in range(num_links):
        # Link
        link = pin.Link(
            name=f"link_{i+1}",
            mass=2.0 - i*0.2,  # Decreasing mass
            center_of_mass=np.array([0.2, 0, 0]),
            inertia_tensor=np.diag([0.1, 0.1, 0.05])
        )
        links.append(link)
        
        # Joint
        parent = "base" if i == 0 else f"link_{i}"
        joint = pin.Joint(
            name=f"joint_{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]) if i % 2 == 0 else np.array([0, 1, 0]),
            parent_link=parent,
            child_link=f"link_{i+1}",
            origin_transform=pin.create_transform(
                np.eye(3), 
                np.array([0.4, 0, 0]) if i > 0 else np.array([0, 0, 0.1])
            )
        )
        joints.append(joint)
    
    return pin.create_robot_model(
        name=f"{num_links}dof_arm",
        links=links,
        joints=joints,
        root_link="base"
    )
```

### Parallel Mechanisms

More complex kinematic structures:

```python
def create_parallel_robot():
    """Create a simple parallel mechanism (Stewart platform concept)."""
    links = []
    joints = []
    
    # Base platform
    base = pin.Link(
        name="base_platform",
        mass=20.0,
        center_of_mass=np.array([0, 0, 0.05]),
        inertia_tensor=np.diag([1.0, 1.0, 0.5])
    )
    links.append(base)
    
    # Moving platform
    platform = pin.Link(
        name="moving_platform", 
        mass=5.0,
        center_of_mass=np.array([0, 0, 0.02]),
        inertia_tensor=np.diag([0.2, 0.2, 0.1])
    )
    links.append(platform)
    
    # Actuator legs (simplified to 3 legs)
    leg_positions = [
        np.array([0.3, 0, 0]),
        np.array([-0.15, 0.26, 0]),
        np.array([-0.15, -0.26, 0])
    ]
    
    for i, pos in enumerate(leg_positions):
        # Actuator link
        actuator = pin.Link(
            name=f"actuator_{i+1}",
            mass=1.0,
            center_of_mass=np.array([0, 0, 0.25]),
            inertia_tensor=np.diag([0.05, 0.05, 0.01])
        )
        links.append(actuator)
        
        # Base connection (prismatic)
        base_joint = pin.Joint(
            name=f"actuator_{i+1}_base",
            joint_type=pin.JointType.PRISMATIC,
            axis=np.array([0, 0, 1]),
            parent_link="base_platform",
            child_link=f"actuator_{i+1}",
            origin_transform=pin.create_transform(np.eye(3), pos),
            position_limit_lower=0.2,
            position_limit_upper=0.8
        )
        joints.append(base_joint)
        
        # Platform connection (fixed for simplicity)
        platform_joint = pin.Joint(
            name=f"actuator_{i+1}_platform",
            joint_type=pin.JointType.FIXED,
            parent_link=f"actuator_{i+1}",
            child_link="moving_platform" if i == 0 else f"virtual_connection_{i}",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, 0.5]))
        )
        joints.append(platform_joint)
        
        # Add virtual connection links for other legs
        if i > 0:
            virtual_link = pin.Link(name=f"virtual_connection_{i}", mass=0.001)
            links.append(virtual_link)
    
    return pin.create_robot_model(
        name="parallel_robot",
        links=links,
        joints=joints,
        root_link="base_platform"
    )
```

### Branched Structures

Robots with multiple arms or legs:

```python
def create_humanoid_torso():
    """Create humanoid upper body with two arms."""
    links = []
    joints = []
    
    # Torso
    torso = pin.Link(
        name="torso",
        mass=30.0,
        center_of_mass=np.array([0, 0, 0.25]),
        inertia_tensor=np.diag([2.0, 1.5, 1.0])
    )
    links.append(torso)
    
    # Create both arms
    for side in ["left", "right"]:
        side_multiplier = -1 if side == "left" else 1
        
        # Shoulder offset
        shoulder_pos = np.array([0, side_multiplier * 0.25, 0.4])
        
        # Upper arm
        upper_arm = pin.Link(
            name=f"{side}_upper_arm",
            mass=3.0,
            center_of_mass=np.array([0, 0, -0.15]),
            inertia_tensor=np.diag([0.1, 0.1, 0.02])
        )
        links.append(upper_arm)
        
        # Forearm
        forearm = pin.Link(
            name=f"{side}_forearm",
            mass=2.0,
            center_of_mass=np.array([0, 0, -0.12]),
            inertia_tensor=np.diag([0.05, 0.05, 0.01])
        )
        links.append(forearm)
        
        # Shoulder joint
        shoulder = pin.Joint(
            name=f"{side}_shoulder",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),
            parent_link="torso",
            child_link=f"{side}_upper_arm",
            origin_transform=pin.create_transform(np.eye(3), shoulder_pos),
            position_limit_lower=-np.pi,
            position_limit_upper=np.pi
        )
        joints.append(shoulder)
        
        # Elbow joint
        elbow = pin.Joint(
            name=f"{side}_elbow",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 1, 0]),
            parent_link=f"{side}_upper_arm",
            child_link=f"{side}_forearm",
            origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, -0.3])),
            position_limit_lower=0.0,
            position_limit_upper=np.pi
        )
        joints.append(elbow)
    
    return pin.create_robot_model(
        name="humanoid_torso",
        links=links,
        joints=joints,
        root_link="torso"
    )
```

## Mass Properties

### Calculating Inertia from Geometry

For complex shapes, combine simple geometries:

```python
def composite_inertia(components):
    """
    Combine inertia tensors of multiple components using the parallel axis theorem.

    The parallel axis theorem states that for a rigid body with inertia tensor I_cm
    about its center of mass, the inertia tensor about a point displaced by vector d is:

    I_new = I_cm + m(‖d‖²I - d⊗d)

    Where:
    - I_cm is the inertia tensor about the center of mass
    - m is the mass of the body
    - d is the displacement vector
    - ‖d‖² is the squared magnitude of d
    - d⊗d is the outer product of d with itself

    Args:
        components: List of (mass, com, inertia, position) tuples
    """
    total_mass = sum(comp[0] for comp in components)
    
    # Calculate combined center of mass
    weighted_com = np.zeros(3)
    for mass, com, _, pos in components:
        weighted_com += mass * (com + pos)
    combined_com = weighted_com / total_mass
    
    # Calculate combined inertia using parallel axis theorem
    combined_inertia = np.zeros((3, 3))
    for mass, com, inertia, pos in components:
        # Component COM in combined frame
        component_com = com + pos
        
        # Distance from combined COM
        d = component_com - combined_com
        
        # Parallel axis theorem: I = I_cm + m * (d²I - d⊗d)
        # Mathematical form: I_new = I_cm + m(‖d‖²I - d⊗d)
        d_squared = np.dot(d, d)
        d_outer = np.outer(d, d)
        parallel_axis = mass * (d_squared * np.eye(3) - d_outer)
        
        combined_inertia += inertia + parallel_axis
    
    return total_mass, combined_com, combined_inertia

# Example: Robot arm link with motor
def create_actuated_link():
    """Create link with embedded motor."""
    # Link structure (aluminum tube)
    link_mass = 2.0
    link_com = np.array([0.25, 0, 0])
    link_inertia = cylinder_inertia(link_mass, 0.03, 0.5)
    
    # Motor (steel cylinder)
    motor_mass = 1.5
    motor_com = np.array([0, 0, 0])
    motor_inertia = cylinder_inertia(motor_mass, 0.05, 0.1)
    motor_position = np.array([0.1, 0, 0])
    
    # Combine properties
    components = [
        (link_mass, link_com, link_inertia, np.zeros(3)),
        (motor_mass, motor_com, motor_inertia, motor_position)
    ]
    
    total_mass, total_com, total_inertia = composite_inertia(components)
    
    return pin.Link(
        name="actuated_link",
        mass=total_mass,
        center_of_mass=total_com,
        inertia_tensor=total_inertia
    )
```

### Scaling Mass Properties

For different robot sizes:

```python
def scale_robot_mass_properties(robot, scale_factor):
    """
    Scale robot mass properties for different sizes.
    
    Scaling laws:
    - Length: scale_factor
    - Area: scale_factor²
    - Volume/Mass: scale_factor³
    - Inertia: scale_factor⁵
    """
    scaled_links = []
    
    for link in robot.links:
        if link.has_inertia:
            scaled_link = pin.Link(
                name=link.name,
                mass=link.mass * scale_factor**3,
                center_of_mass=link.center_of_mass * scale_factor,
                inertia_tensor=link.inertia_tensor * scale_factor**5
            )
        else:
            scaled_link = link
        
        scaled_links.append(scaled_link)
    
    # Note: Would also need to scale joint origins and limits
    return scaled_links
```

## Validation and Debugging

### Model Validation

Always validate your robot models:

```python
def validate_robot_model(robot):
    """Comprehensive robot model validation."""
    issues = []
    
    # Check basic properties
    if robot.num_dof == 0:
        issues.append("Robot has no degrees of freedom")
    
    if robot.num_links < 2:
        issues.append("Robot should have at least 2 links")
    
    # Check mass properties
    total_mass = sum(link.mass for link in robot.links if link.has_inertia)
    if total_mass <= 0:
        issues.append("Robot has no mass")
    
    # Check for duplicate names
    link_names = [link.name for link in robot.links]
    if len(link_names) != len(set(link_names)):
        issues.append("Duplicate link names found")
    
    joint_names = [joint.name for joint in robot.joints]
    if len(joint_names) != len(set(joint_names)):
        issues.append("Duplicate joint names found")
    
    # Check kinematic connectivity
    for joint in robot.joints:
        if joint.parent_link not in link_names:
            issues.append(f"Joint {joint.name} references unknown parent link")
        if joint.child_link not in link_names:
            issues.append(f"Joint {joint.name} references unknown child link")
    
    # Check for kinematic loops (simplified)
    # In a tree structure, num_joints should equal num_links - 1
    if len(robot.joints) != len(robot.links) - 1:
        issues.append("Possible kinematic loop or disconnected links")
    
    # Check inertia tensors
    for link in robot.links:
        if link.has_inertia:
            eigenvals = np.linalg.eigvals(link.inertia_tensor)
            if not np.all(eigenvals > 0):
                issues.append(f"Link {link.name} has invalid inertia tensor")
    
    return issues

# Usage
robot = create_serial_arm(6)
validation_issues = validate_robot_model(robot)

if validation_issues:
    print("Model validation issues:")
    for issue in validation_issues:
        print(f"  - {issue}")
else:
    print("Robot model validation passed!")
```

### Debugging Tips

Common issues and solutions:

```python
# 1. Check joint configuration validity
def debug_joint_config(robot, q):
    """Debug joint configuration issues."""
    if len(q) != robot.num_dof:
        print(f"Wrong config size: expected {robot.num_dof}, got {len(q)}")
        return False
    
    for i, (joint_name, angle) in enumerate(zip(robot.actuated_joint_names, q)):
        joint = robot.get_joint(joint_name)
        if joint:
            if angle < joint.position_limit_lower:
                print(f"Joint {joint_name} below lower limit: {angle} < {joint.position_limit_lower}")
                return False
            if angle > joint.position_limit_upper:
                print(f"Joint {joint_name} above upper limit: {angle} > {joint.position_limit_upper}")
                return False
    
    return True

# 2. Visualize mass distribution
def analyze_mass_distribution(robot):
    """Analyze robot mass distribution."""
    print("Mass distribution analysis:")
    
    total_mass = 0
    for link in robot.links:
        if link.has_inertia:
            print(f"  {link.name}: {link.mass:.2f} kg")
            total_mass += link.mass
    
    print(f"Total mass: {total_mass:.2f} kg")
    
    # Check for unrealistic mass ratios
    masses = [link.mass for link in robot.links if link.has_inertia]
    if masses:
        max_mass = max(masses)
        min_mass = min(masses)
        ratio = max_mass / min_mass
        
        if ratio > 100:
            print(f"Warning: Large mass ratio ({ratio:.1f}:1) may cause numerical issues")

# 3. Test forward kinematics
def test_forward_kinematics(robot, num_tests=100):
    """Test forward kinematics with random configurations."""
    print(f"Testing forward kinematics with {num_tests} random configurations...")
    
    failures = 0
    for i in range(num_tests):
        # Generate random valid configuration
        q = []
        for joint_name in robot.actuated_joint_names:
            joint = robot.get_joint(joint_name)
            if joint:
                q_min = joint.position_limit_lower
                q_max = joint.position_limit_upper
                q.append(np.random.uniform(q_min, q_max))
        
        q = np.array(q)
        
        try:
            kinematic_state = pin.compute_forward_kinematics(robot, q)
            # Check for NaN or infinite values
            for link_name, transform in kinematic_state.link_transforms.items():
                if np.any(np.isnan(transform.translation)) or np.any(np.isinf(transform.translation)):
                    failures += 1
                    break
        except Exception:
            failures += 1
    
    success_rate = (num_tests - failures) / num_tests
    print(f"Forward kinematics success rate: {success_rate:.1%}")
    
    if success_rate < 0.95:
        print("Warning: Low success rate indicates model issues")
```

## Next Steps

Now that you understand robot modeling, explore:

1. **[Kinematics](03_kinematics.md)**: Forward and inverse kinematics
2. **[Dynamics](04_dynamics.md)**: Robot dynamics and control
3. **[File Formats](06_file_formats.md)**: Loading robots from URDF/MJCF
4. **Advanced Examples**: Study complex robot examples in the examples directory

## Summary

Key takeaways for robot modeling:

- **Accurate mass properties** are crucial for realistic dynamics
- **Proper joint limits** prevent unrealistic configurations  
- **Validation** catches modeling errors early
- **Modular design** makes complex robots manageable
- **Consistent units** prevent scaling issues

With these techniques, you can model any robot from simple arms to complex humanoids!
