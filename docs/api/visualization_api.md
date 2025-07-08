# Visualization API Reference

This document provides comprehensive API documentation for py-pinocchio's visualization capabilities, including 2D/3D plotting, animation, and workspace analysis.

## Table of Contents

1. [Robot Visualizer](#robot-visualizer)
2. [Static Plotting](#static-plotting)
3. [Animation](#animation)
4. [Workspace Analysis](#workspace-analysis)
5. [Jacobian Analysis](#jacobian-analysis)

## Robot Visualizer

### `RobotVisualizer`

Main class for robot visualization and animation.

```python
class RobotVisualizer:
    """Interactive robot visualization and animation."""
    
    def __init__(self, robot: RobotModel, figure_size: Tuple[int, int] = (10, 8)):
        """Initialize visualizer for a robot model."""
```

#### Constructor Parameters

- **`robot`** (`RobotModel`): Robot model to visualize
- **`figure_size`** (`Tuple[int, int]`, optional): Figure size in inches. Default: `(10, 8)`

#### Methods

##### `plot_configuration(joint_positions: np.ndarray, title: str = "") -> None`

Plot robot in a specific configuration.

**Parameters:**
- `joint_positions`: Joint angles/positions
- `title`: Plot title (optional)

##### `animate_trajectory(trajectory: np.ndarray, dt: float = 0.1) -> None`

Animate robot following a trajectory.

**Parameters:**
- `trajectory`: Array of joint positions over time (shape: `[time_steps, num_dof]`)
- `dt`: Time step between frames in seconds

## Static Plotting

### `plot_robot_2d(robot: RobotModel, joint_positions: np.ndarray, **kwargs) -> None`

Plot robot in 2D configuration.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint configuration
- `**kwargs`: Additional matplotlib parameters

**Keyword Arguments:**
- `title` (`str`): Plot title
- `show_frames` (`bool`): Show coordinate frames. Default: `True`
- `show_labels` (`bool`): Show link/joint labels. Default: `True`
- `link_color` (`str`): Link color. Default: `'blue'`
- `joint_color` (`str`): Joint color. Default: `'red'`
- `frame_scale` (`float`): Coordinate frame scale. Default: `0.1`

**Example:**
```python
import py_pinocchio as pin
import numpy as np

# Create robot and configuration
robot = create_2dof_robot()
q = np.array([np.pi/4, np.pi/6])

# Plot robot
pin.plot_robot_2d(robot, q, title="2-DOF Robot", show_frames=True)
```

### `plot_robot_3d(robot: RobotModel, joint_positions: np.ndarray, **kwargs) -> None`

Plot robot in 3D configuration.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint configuration
- `**kwargs`: Additional plotting parameters

**Keyword Arguments:**
- `title` (`str`): Plot title
- `show_frames` (`bool`): Show coordinate frames. Default: `True`
- `show_labels` (`bool`): Show link/joint labels. Default: `True`
- `link_color` (`str`): Link color. Default: `'blue'`
- `joint_color` (`str`): Joint color. Default: `'red'`
- `frame_scale` (`float`): Coordinate frame scale. Default: `0.1`
- `view_angle` (`Tuple[float, float]`): Viewing angles (elevation, azimuth)

## Animation

### `animate_robot_motion(robot: RobotModel, trajectory: np.ndarray, **kwargs) -> None`

Create animated visualization of robot motion.

**Parameters:**
- `robot`: Robot model
- `trajectory`: Joint trajectory (shape: `[time_steps, num_dof]`)
- `**kwargs`: Animation parameters

**Keyword Arguments:**
- `dt` (`float`): Time step between frames. Default: `0.1`
- `title` (`str`): Animation title
- `save_path` (`str`): Path to save animation (optional)
- `fps` (`int`): Frames per second for saved animation. Default: `10`
- `show_trail` (`bool`): Show end-effector trail. Default: `False`
- `trail_length` (`int`): Length of trail in frames. Default: `50`

**Example:**
```python
# Generate trajectory
time_steps = 100
trajectory = np.zeros((time_steps, robot.num_dof))
for i in range(time_steps):
    t = i * 0.1
    trajectory[i] = [np.sin(t), np.cos(t)]

# Animate
pin.animate_robot_motion(robot, trajectory, dt=0.1, show_trail=True)
```

## Workspace Analysis

### `plot_workspace(robot: RobotModel, end_effector_link: str, **kwargs) -> np.ndarray`

Plot and analyze robot workspace.

**Parameters:**
- `robot`: Robot model
- `end_effector_link`: Name of end-effector link
- `**kwargs`: Workspace analysis parameters

**Keyword Arguments:**
- `num_samples` (`int`): Number of random samples. Default: `1000`
- `joint_limits` (`bool`): Respect joint limits. Default: `True`
- `plot_type` (`str`): Plot type ('2d', '3d', 'both'). Default: `'2d'`
- `color_by` (`str`): Color coding ('reachability', 'manipulability'). Default: `'reachability'`
- `alpha` (`float`): Point transparency. Default: `0.6`

**Returns:**
- `np.ndarray`: Workspace points (shape: `[num_samples, 3]`)

**Example:**
```python
# Analyze workspace
workspace_points = pin.plot_workspace(
    robot, 
    "end_effector",
    num_samples=2000,
    plot_type='both',
    color_by='manipulability'
)

print(f"Workspace volume: {estimate_workspace_volume(workspace_points):.3f} m³")
```

## Jacobian Analysis

### `plot_jacobian_analysis(robot: RobotModel, joint_positions: np.ndarray, end_effector_link: str, **kwargs) -> Dict`

Visualize Jacobian properties and singularity analysis.

**Parameters:**
- `robot`: Robot model
- `joint_positions`: Joint configuration
- `end_effector_link`: End-effector link name
- `**kwargs`: Analysis parameters

**Keyword Arguments:**
- `plot_ellipsoid` (`bool`): Plot manipulability ellipsoid. Default: `True`
- `plot_nullspace` (`bool`): Plot null space vectors. Default: `True`
- `ellipsoid_scale` (`float`): Ellipsoid scaling factor. Default: `1.0`
- `vector_scale` (`float`): Vector scaling factor. Default: `0.1`

**Returns:**
- `Dict`: Analysis results containing:
  - `singular_values`: Jacobian singular values
  - `manipulability`: Manipulability measure
  - `condition_number`: Condition number
  - `is_singular`: Singularity flag
  - `null_space`: Null space basis vectors

**Example:**
```python
# Analyze Jacobian at specific configuration
q = np.array([np.pi/3, np.pi/4])
analysis = pin.plot_jacobian_analysis(
    robot, q, "end_effector",
    plot_ellipsoid=True,
    plot_nullspace=True
)

print(f"Manipulability: {analysis['manipulability']:.4f}")
print(f"Condition number: {analysis['condition_number']:.2f}")
print(f"Singular: {analysis['is_singular']}")
```

## Visualization Utilities

### Color Schemes

The visualization module provides several predefined color schemes:

```python
# Available color schemes
ROBOT_COLORS = {
    'default': {'links': 'blue', 'joints': 'red', 'frames': 'black'},
    'industrial': {'links': 'gray', 'joints': 'orange', 'frames': 'green'},
    'academic': {'links': 'navy', 'joints': 'crimson', 'frames': 'darkgreen'}
}
```

### Configuration

Global visualization settings can be configured:

```python
import py_pinocchio as pin
import matplotlib.pyplot as plt

# Set matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['animation.writer'] = 'ffmpeg'

# Note: Global visualization settings are configured through matplotlib
```

## Requirements

The visualization module requires:
- `matplotlib >= 3.5.0`
- `numpy >= 1.20.0`

Optional dependencies for enhanced features:
- `ffmpeg` (for saving animations)
- `pillow` (for image processing)

## Notes

- All visualization functions are optional and gracefully handle missing matplotlib
- 3D plotting uses matplotlib's `mplot3d` toolkit
- Animations can be saved in various formats (MP4, GIF, etc.)
- Interactive features work best in Jupyter notebooks or interactive Python sessions
