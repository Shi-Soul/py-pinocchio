# Core API Reference

This document provides comprehensive API documentation for py-pinocchio's core functionality.

## Table of Contents

1. [Robot Model](#robot-model)
2. [Links and Joints](#links-and-joints)
3. [Transformations](#transformations)
4. [Kinematics](#kinematics)
5. [Dynamics](#dynamics)
6. [Jacobians](#jacobians)

## Robot Model

### `RobotModel`

The main class representing a complete robot.

```python
@dataclass(frozen=True)
class RobotModel:
    """Immutable robot model representation."""
    name: str
    links: Tuple[Link, ...]
    joints: Tuple[Joint, ...]
    root_link: str
```

#### Properties

- **`name`** (`str`): Robot model name
- **`links`** (`Tuple[Link, ...]`): Tuple of all links in the robot
- **`joints`** (`Tuple[Joint, ...]`): Tuple of all joints in the robot
- **`root_link`** (`str`): Name of the root (base) link
- **`num_links`** (`int`): Number of links in the model
- **`num_joints`** (`int`): Number of joints in the model
- **`num_dof`** (`int`): Total degrees of freedom (actuated joints only)
- **`actuated_joint_names`** (`List[str]`): Names of actuated joints in kinematic order

#### Methods

##### `get_link(name: str) -> Optional[Link]`

Get link by name.

**Parameters:**
- `name`: Link name

**Returns:**
- Link object if found, None otherwise

**Example:**
```python
base_link = robot.get_link("base")
if base_link:
    print(f"Base mass: {base_link.mass} kg")
```

##### `get_joint(name: str) -> Optional[Joint]`

Get joint by name.

**Parameters:**
- `name`: Joint name

**Returns:**
- Joint object if found, None otherwise

##### `is_valid_joint_configuration(q: np.ndarray) -> bool`

Check if joint configuration is within limits.

**Parameters:**
- `q`: Joint positions array (size = num_dof)

**Returns:**
- True if configuration is valid, False otherwise

**Example:**
```python
q = np.array([0.5, -0.3, 1.2])
if robot.is_valid_joint_configuration(q):
    print("Configuration is valid")
```

### `create_robot_model()`

Factory function to create robot models.

```python
def create_robot_model(
    name: str, 
    links: List[Link], 
    joints: List[Joint], 
    root_link: str = ""
) -> RobotModel
```

**Parameters:**
- `name`: Robot model name
- `links`: List of links
- `joints`: List of joints
- `root_link`: Name of root link (base)

**Returns:**
- Immutable RobotModel object

**Example:**
```python
robot = pin.create_robot_model(
    name="my_robot",
    links=[base, link1, link2],
    joints=[joint1, joint2],
    root_link="base"
)
```

## Links and Joints

### `Link`

Represents a rigid body in the robot.

```python
@dataclass(frozen=True)
class Link:
    """Immutable link representation."""
    name: str
    mass: float = 0.0
    center_of_mass: Vector3 = field(default_factory=lambda: np.zeros(3))
    inertia_tensor: Matrix3x3 = field(default_factory=lambda: np.eye(3))
```

#### Properties

- **`name`** (`str`): Link name (must be unique)
- **`mass`** (`float`): Link mass in kg
- **`center_of_mass`** (`Vector3`): Center of mass position in link frame (m)
- **`inertia_tensor`** (`Matrix3x3`): $3 \times 3$ inertia tensor in link frame (kg⋅m²)
- **`has_inertia`** (`bool`): True if link has non-zero mass

**Example:**
```python
link = pin.Link(
    name="upper_arm",
    mass=2.5,
    center_of_mass=np.array([0.15, 0.0, 0.0]),
    inertia_tensor=np.diag([0.1, 0.1, 0.02])
)
```

### `Joint`

Represents a connection between two links.

```python
@dataclass(frozen=True)
class Joint:
    """Immutable joint representation."""
    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    axis: Vector3 = field(default_factory=lambda: np.array([0, 0, 1]))
    origin_transform: Transform = field(default_factory=create_identity_transform)
    position_limit_lower: float = -np.inf
    position_limit_upper: float = np.inf
```

#### Properties

- **`name`** (`str`): Joint name (must be unique)
- **`joint_type`** (`JointType`): Type of joint (REVOLUTE, PRISMATIC, FIXED)
- **`parent_link`** (`str`): Name of parent link
- **`child_link`** (`str`): Name of child link
- **`axis`** (`Vector3`): Joint axis (rotation axis for revolute, translation direction for prismatic)
- **`origin_transform`** (`Transform`): Transform from parent link to joint frame
- **`position_limit_lower`** (`float`): Lower joint limit
- **`position_limit_upper`** (`float`): Upper joint limit
- **`is_actuated`** (`bool`): True if joint is actuated (not fixed)
- **`degrees_of_freedom`** (`int`): Number of DOF (1 for revolute/prismatic, 0 for fixed)

### `JointType`

Enumeration of supported joint types.

```python
class JointType(Enum):
    """Supported joint types."""
    REVOLUTE = "revolute"      # Rotational joint
    PRISMATIC = "prismatic"    # Linear joint
    FIXED = "fixed"           # Rigid connection
```

**Example:**
```python
joint = pin.Joint(
    name="shoulder_pitch",
    joint_type=pin.JointType.REVOLUTE,
    axis=np.array([0, 1, 0]),  # Y-axis rotation
    parent_link="torso",
    child_link="upper_arm",
    position_limit_lower=-np.pi/2,
    position_limit_upper=np.pi/2
)
```

## Transformations

### `Transform`

Immutable 3D transformation representation.

```python
class Transform(NamedTuple):
    """3D transformation with rotation and translation."""
    rotation: Matrix3x3
    translation: Vector3
```

#### Properties

- **`rotation`** (`Matrix3x3`): $3 \times 3$ rotation matrix
- **`translation`** (`Vector3`): 3D translation vector

### Transformation Functions

#### `create_identity_transform() -> Transform`

Create identity transformation.

**Returns:**
- Identity transform (no rotation, no translation)

#### `create_transform(rotation: Matrix3x3, translation: Vector3) -> Transform`

Create transformation from rotation matrix and translation vector.

**Parameters:**
- `rotation`: $3 \times 3$ rotation matrix
- `translation`: 3D translation vector

**Returns:**
- Transform object

**Raises:**
- `ValueError`: If rotation or translation are invalid

#### `compose_transforms(transform1: Transform, transform2: Transform) -> Transform`

Compose two transformations.

**Parameters:**
- `transform1`: First transformation (applied second)
- `transform2`: Second transformation (applied first)

**Returns:**
- Composed transformation

**Mathematical Operation:**
```
T_result = T1 * T2
Point transformation: p' = T1(T2(p))
```

#### `invert_transform(transform: Transform) -> Transform`

Compute inverse of transformation.

**Parameters:**
- `transform`: Transform to invert

**Returns:**
- Inverse transformation

**Mathematical Operation:**
For transform $\mathbf{T} = (\mathbf{R}, \mathbf{t})$:
$$\mathbf{T}^{-1} = (\mathbf{R}^T, -\mathbf{R}^T \mathbf{t})$$

#### `transform_point(transform: Transform, point: np.ndarray) -> np.ndarray`

Apply transformation to a 3D point.

**Parameters:**
- `transform`: Transformation to apply
- `point`: 3D point to transform

**Returns:**
- Transformed point

**Mathematical Operation:**
$$\mathbf{p}' = \mathbf{R} \mathbf{p} + \mathbf{t}$$

#### `transform_vector(transform: Transform, vector: np.ndarray) -> np.ndarray`

Apply only rotation part of transformation to a vector.

**Parameters:**
- `transform`: Transformation (only rotation used)
- `vector`: 3D vector to transform

**Returns:**
- Transformed vector

**Mathematical Operation:**
$$\mathbf{v}' = \mathbf{R} \mathbf{v}$$

## Kinematics

### `KinematicState`

Result of forward kinematics computation.

```python
@dataclass(frozen=True)
class KinematicState:
    """Complete kinematic state of robot."""
    link_transforms: Dict[str, Transform]
    joint_transforms: Dict[str, Transform]
```

#### Properties

- **`link_transforms`** (`Dict[str, Transform]`): World frame transforms for all links
- **`joint_transforms`** (`Dict[str, Transform]`): World frame transforms for all joints

### Forward Kinematics Functions

#### `compute_forward_kinematics(robot: RobotModel, joint_positions: np.ndarray) -> KinematicState`

Compute forward kinematics for entire robot.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint positions for actuated joints

**Returns:**
- KinematicState with all link and joint transforms

**Raises:**
- `ValueError`: If joint_positions has wrong size or values out of limits

**Example:**
```python
q = np.array([0.5, -0.3, 1.2])
kinematic_state = pin.compute_forward_kinematics(robot, q)

# Access link transforms
end_effector_transform = kinematic_state.link_transforms["end_effector"]
```

#### `get_link_transform(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> Transform`

Get transform for a specific link.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint positions
- `link_name`: Name of link to get transform for

**Returns:**
- Transform from world frame to link frame

**Raises:**
- `ValueError`: If link doesn't exist

#### `get_link_position(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray`

Get position of a specific link's origin.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint positions
- `link_name`: Name of link

**Returns:**
- 3D position vector in world frame

#### `get_link_orientation(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray`

Get orientation of a specific link.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint positions
- `link_name`: Name of link

**Returns:**
- $3 \times 3$ rotation matrix representing link orientation in world frame

## Dynamics

### Mass Matrix

#### `compute_mass_matrix(robot: RobotModel, joint_positions: np.ndarray) -> np.ndarray`

Compute joint-space mass matrix.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions

**Returns:**
- $n \times n$ mass matrix where $n$ = num_dof

**Properties:**
- Symmetric: M = M^T
- Positive definite: all eigenvalues > 0
- Configuration dependent

**Mathematical Relation:**
$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

### Gravity Forces

#### `compute_gravity_vector(robot: RobotModel, joint_positions: np.ndarray) -> np.ndarray`

Compute gravity compensation torques.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions

**Returns:**
- $n \times 1$ gravity vector where $n$ = num_dof

**Note:** Assumes gravity vector [0, 0, -9.81] m/s²

### Coriolis Forces

#### `compute_coriolis_forces(robot: RobotModel, joint_positions: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray`

Compute Coriolis and centrifugal forces.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions
- `joint_velocities`: Current joint velocities

**Returns:**
- $n \times 1$ Coriolis force vector where $n$ = num_dof

### Forward Dynamics

#### `compute_forward_dynamics(robot: RobotModel, joint_positions: np.ndarray, joint_velocities: np.ndarray, joint_torques: np.ndarray) -> np.ndarray`

Compute joint accelerations from applied torques.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions
- `joint_velocities`: Current joint velocities
- `joint_torques`: Applied joint torques

**Returns:**
- $n \times 1$ joint acceleration vector where $n$ = num_dof

**Mathematical Operation:**
$$\ddot{\mathbf{q}} = \mathbf{M}(\mathbf{q})^{-1} \left[\boldsymbol{\tau} - \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} - \mathbf{g}(\mathbf{q})\right]$$

### Inverse Dynamics

#### `compute_inverse_dynamics(robot: RobotModel, joint_positions: np.ndarray, joint_velocities: np.ndarray, joint_accelerations: np.ndarray) -> np.ndarray`

Compute required torques for desired motion.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions
- `joint_velocities`: Current joint velocities
- `joint_accelerations`: Desired joint accelerations

**Returns:**
- $n \times 1$ required torque vector where $n$ = num_dof

**Mathematical Operation:**
$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

## Jacobians

### `compute_geometric_jacobian(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray`

Compute geometric Jacobian for a link.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions
- `link_name`: Name of link to compute Jacobian for

**Returns:**
- $6 \times n$ Jacobian matrix where $n$ = num_dof

**Structure:**
```
J = [J_angular]  (3×n)
    [J_linear ]  (3×n)
```

**Mathematical Relation:**
$$\mathbf{v} = \mathbf{J} \dot{\mathbf{q}}$$
where $\mathbf{v} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}$ is 6D spatial velocity

### `compute_analytical_jacobian(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray`

Compute analytical Jacobian for a link.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Current joint positions
- `link_name`: Name of link to compute Jacobian for

**Returns:**
- $6 \times n$ analytical Jacobian matrix where $n$ = num_dof

**Note:** Analytical Jacobian relates joint velocities to end-effector linear and angular velocities in a more intuitive coordinate system.

## Type Definitions

### Common Types

```python
# Basic numpy array types
Vector3: TypeAlias = np.ndarray      # 3D vector
Matrix3x3: TypeAlias = np.ndarray    # $3 \times 3$ matrix
Matrix4x4: TypeAlias = np.ndarray    # $4 \times 4$ homogeneous transformation
Matrix6x6: TypeAlias = np.ndarray    # $6 \times 6$ spatial matrix

# Joint configuration types
JointPositions: TypeAlias = np.ndarray    # Array of joint positions
JointVelocities: TypeAlias = np.ndarray   # Array of joint velocities
JointAccelerations: TypeAlias = np.ndarray # Array of joint accelerations
JointTorques: TypeAlias = np.ndarray      # Array of joint torques

# String identifiers
LinkName: TypeAlias = str
JointName: TypeAlias = str
```

### Validation Functions

#### `validate_vector3(v: Any) -> Vector3`

Validate and convert to 3D vector.

#### `validate_matrix3x3(m: Any) -> Matrix3x3`

Validate and convert to 3×3 matrix.

#### `validate_joint_positions(q: Any, expected_dof: int) -> JointPositions`

Validate joint positions array.

## Error Handling

All functions may raise the following exceptions:

- **`ValueError`**: Invalid input parameters (wrong size, out of bounds, etc.)
- **`KeyError`**: Referenced link or joint not found
- **`RuntimeError`**: Computation failed (singular matrices, etc.)

## Usage Examples

### Complete Robot Analysis

```python
import numpy as np
import py_pinocchio as pin

# Create robot (see robot modeling tutorial)
robot = create_my_robot()

# Define configuration
q = np.array([0.1, 0.2, -0.3])
qd = np.array([0.0, 0.1, 0.0])
tau = np.array([1.0, 0.5, 0.2])

# Forward kinematics
ee_pos = pin.get_link_position(robot, q, "end_effector")
ee_rot = pin.get_link_orientation(robot, q, "end_effector")

# Jacobian
J = pin.compute_geometric_jacobian(robot, q, "end_effector")

# Dynamics
M = pin.compute_mass_matrix(robot, q)
g = pin.compute_gravity_vector(robot, q)
qdd = pin.compute_forward_dynamics(robot, q, qd, tau)

print(f"End-effector position: {ee_pos}")
print(f"Joint accelerations: {qdd}")
```

This API reference provides the foundation for all robotics computations in py-pinocchio. For more advanced usage, see the tutorials and examples.
