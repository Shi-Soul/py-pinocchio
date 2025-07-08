# Getting Started with py-pinocchio

Welcome to py-pinocchio! This tutorial will guide you through the basics of using py-pinocchio for robotics applications. By the end of this tutorial, you'll understand how to create robot models, compute forward kinematics, and perform basic analysis.

## What is py-pinocchio?

py-pinocchio is an educational implementation of rigid body dynamics algorithms for robotics. It's designed to help you understand the fundamental concepts of robot kinematics and dynamics through clear, well-documented code.

### Key Features

- **Educational Focus**: Code optimized for understanding over performance
- **Functional Programming**: Pure functions with immutable data structures
- **Complete Dynamics**: Forward and inverse dynamics with spatial algebra
- **Multi-Format Support**: URDF and MJCF file parsing
- **Visualization**: 2D/3D robot plotting and animation
- **Type Safety**: Full type annotations for better development experience

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy 1.20.0 or higher
- matplotlib 3.5.0 or higher (for visualization)

### Install from Source

```bash
git clone https://github.com/your-repo/py-pinocchio
cd py-pinocchio
pip install -e .
```

### Verify Installation

```python
import py_pinocchio as pin
print(f"py-pinocchio version: {pin.__version__}")
```

## Basic Concepts

### Robot Models

In py-pinocchio, a robot is represented as a collection of **links** connected by **joints**:

- **Links**: Rigid bodies with mass, inertia, and geometry
- **Joints**: Connections that allow relative motion between links
- **Robot Model**: Complete kinematic and dynamic description

### Coordinate Frames

- **World Frame**: Fixed reference frame
- **Link Frames**: Attached to each link
- **Joint Frames**: Located at joint axes

### Transformations

3D transformations are represented using:
- **Rotation matrices**: $3 \times 3$ orthogonal matrices
- **Translation vectors**: 3D position vectors
- **Transform objects**: Combined rotation and translation

## Your First Robot

Let's create a simple 2-DOF planar robot arm:

```python
import numpy as np
import py_pinocchio as pin

# Create links
base = pin.Link(
    name="base",
    mass=1.0,
    center_of_mass=np.array([0.0, 0.0, 0.0])
)

link1 = pin.Link(
    name="link1", 
    mass=2.0,
    center_of_mass=np.array([0.25, 0.0, 0.0]),  # COM at middle of link
    inertia_tensor=np.diag([0.1, 0.1, 0.05])
)

link2 = pin.Link(
    name="link2",
    mass=1.5,
    center_of_mass=np.array([0.15, 0.0, 0.0]),
    inertia_tensor=np.diag([0.05, 0.05, 0.02])
)

# Create joints
joint1 = pin.Joint(
    name="joint1",
    joint_type=pin.JointType.REVOLUTE,
    axis=np.array([0, 0, 1]),  # Rotation about Z-axis
    parent_link="base",
    child_link="link1",
    origin_transform=pin.create_transform(np.eye(3), np.array([0, 0, 0]))
)

joint2 = pin.Joint(
    name="joint2",
    joint_type=pin.JointType.REVOLUTE,
    axis=np.array([0, 0, 1]),
    parent_link="link1", 
    child_link="link2",
    origin_transform=pin.create_transform(np.eye(3), np.array([0.5, 0, 0]))  # 0.5m offset
)

# Create robot model
robot = pin.create_robot_model(
    name="simple_arm",
    links=[base, link1, link2],
    joints=[joint1, joint2],
    root_link="base"
)

print(f"Created robot with {robot.num_dof} degrees of freedom")
```

## Forward Kinematics

Forward kinematics computes the position and orientation of robot links given joint angles:

```python
# Define joint configuration
q = np.array([np.pi/4, np.pi/6])  # 45° and 30°

# Compute forward kinematics
kinematic_state = pin.compute_forward_kinematics(robot, q)

# Get end-effector position
end_effector_pos = pin.get_link_position(robot, q, "link2")
print(f"End-effector position: {end_effector_pos}")

# Get end-effector orientation
end_effector_rot = pin.get_link_orientation(robot, q, "link2")
print(f"End-effector rotation matrix:\n{end_effector_rot}")
```

## Jacobian Computation

The Jacobian relates joint velocities to end-effector velocities:

```python
# Compute geometric Jacobian
jacobian = pin.compute_geometric_jacobian(robot, q, "link2")
print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian:\n{jacobian}")

# The Jacobian is $6 \times 2$ (6D spatial velocity, 2 DOF)
# Top 3 rows: angular velocity components
# Bottom 3 rows: linear velocity components
```

## Robot Dynamics

Robot dynamics describes the relationship between forces/torques and motion. The fundamental equation of motion for a robot is:

$$\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

Where:
- $\mathbf{M}(\mathbf{q})$ is the mass matrix
- $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$ represents Coriolis and centrifugal forces
- $\mathbf{g}(\mathbf{q})$ is the gravity vector
- $\boldsymbol{\tau}$ is the vector of applied joint torques

### Mass Matrix

The mass matrix relates joint accelerations to joint torques:

```python
# Compute mass matrix
M = pin.compute_mass_matrix(robot, q)
print(f"Mass matrix:\n{M}")

# Properties of mass matrix:
# - Symmetric: $\mathbf{M} = \mathbf{M}^T$
# - Positive definite: all eigenvalues > 0
# - Configuration dependent
#
# The mass matrix appears in the equation of motion:
# $\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
```

### Gravity Forces

Gravity affects joint torques:

```python
# Compute gravity vector
g = pin.compute_gravity_vector(robot, q)
print(f"Gravity torques: {g}")
```

### Forward Dynamics

Given joint torques, compute joint accelerations:

```python
# Joint velocities and applied torques
qd = np.array([0.1, -0.2])  # rad/s
tau = np.array([1.0, 0.5])  # Nm

# Compute forward dynamics
qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
print(f"Joint accelerations: {qdd} rad/s²")
```

### Inverse Dynamics

Given desired motion, compute required torques:

```python
# Desired joint accelerations
qdd_desired = np.array([0.5, -0.3])  # rad/s²

# Compute inverse dynamics
tau_required = pin.compute_inverse_dynamics(robot, q, qd, qdd_desired)
print(f"Required torques: {tau_required} Nm")
```

## Working with Transformations

py-pinocchio provides utilities for 3D transformations:

```python
# Create identity transform
T_identity = pin.create_identity_transform()

# Create transform from rotation and translation
rotation = pin.math.utils.rodrigues_formula(np.array([0, 0, 1]), np.pi/4)
translation = np.array([1.0, 0.5, 0.0])
T = pin.create_transform(rotation, translation)

# Transform composition
T_composed = pin.math.transform.compose_transforms(T, T_identity)

# Transform inversion
T_inverse = pin.math.transform.invert_transform(T)

# Transform points
point = np.array([1.0, 0.0, 0.0])
transformed_point = pin.math.transform.transform_point(T, point)
```

## Validation and Error Checking

Always validate your robot configurations:

```python
# Check if joint configuration is valid
q_test = np.array([np.pi, np.pi/2])
is_valid = robot.is_valid_joint_configuration(q_test)
print(f"Configuration valid: {is_valid}")

# Check robot model properties
print(f"Robot name: {robot.name}")
print(f"Number of links: {robot.num_links}")
print(f"Number of joints: {robot.num_joints}")
print(f"Degrees of freedom: {robot.num_dof}")
print(f"Actuated joints: {robot.actuated_joint_names}")
```

## Best Practices

### 1. Use Meaningful Names
```python
# Good
shoulder_joint = pin.Joint(name="shoulder_pitch", ...)

# Avoid
joint1 = pin.Joint(name="j1", ...)
```

### 2. Validate Inputs
```python
def safe_forward_kinematics(robot, q):
    if not robot.is_valid_joint_configuration(q):
        raise ValueError("Invalid joint configuration")
    return pin.compute_forward_kinematics(robot, q)
```

### 3. Handle Exceptions
```python
try:
    jacobian = pin.compute_geometric_jacobian(robot, q, "end_effector")
except Exception as e:
    print(f"Jacobian computation failed: {e}")
```

### 4. Use Type Hints
```python
from typing import List
import numpy as np

def create_robot_links() -> List[pin.Link]:
    """Create robot links with proper type hints."""
    return [
        pin.Link("base", mass=1.0),
        pin.Link("link1", mass=2.0)
    ]
```

## Common Pitfalls

### 1. Incorrect Joint Axes
```python
# Wrong: axis not normalized
joint = pin.Joint(axis=np.array([1, 1, 0]), ...)

# Correct: normalized axis
joint = pin.Joint(axis=np.array([1, 1, 0]) / np.sqrt(2), ...)
```

### 2. Missing Center of Mass
```python
# Wrong: no center of mass specified
link = pin.Link("link1", mass=2.0)

# Correct: specify center of mass
link = pin.Link("link1", mass=2.0, center_of_mass=np.array([0.25, 0, 0]))
```

### 3. Inconsistent Units
```python
# Be consistent with units throughout
# Lengths in meters, masses in kg, angles in radians
```

## Next Steps

Now that you understand the basics, explore these topics:

1. **[Robot Modeling](02_robot_modeling.md)**: Advanced robot creation techniques
2. **[Kinematics](03_kinematics.md)**: Deep dive into forward and inverse kinematics
3. **[Dynamics](04_dynamics.md)**: Understanding robot dynamics and control
4. **[Visualization](05_visualization.md)**: Plotting and animating robots
5. **[File Formats](06_file_formats.md)**: Working with URDF and MJCF files

## Example Scripts

Check out these complete examples:

- `examples/basic_usage.py`: Simple robot creation and analysis
- `examples/legged_robot_example.py`: Bipedal robot modeling
- `examples/multi_dof_arm_example.py`: 6-DOF manipulator
- `examples/visualization_example.py`: Robot visualization

## Getting Help

- **Documentation**: Browse the full documentation
- **Examples**: Study the example scripts
- **API Reference**: Check function signatures and parameters
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions

Happy robotics programming with py-pinocchio! 🤖
