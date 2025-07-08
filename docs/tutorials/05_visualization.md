# Robot Visualization in py-pinocchio

This tutorial covers comprehensive robot visualization techniques, from basic 2D plots to advanced 3D animations. You'll learn how to create publication-quality figures and interactive visualizations.

## Table of Contents

1. [2D Robot Visualization](#2d-robot-visualization)
2. [3D Robot Rendering](#3d-robot-rendering)
3. [Trajectory Visualization](#trajectory-visualization)
4. [Workspace Analysis Plots](#workspace-analysis-plots)
5. [Animation and Interactive Plots](#animation-and-interactive-plots)
6. [Publication-Quality Figures](#publication-quality-figures)

## 2D Robot Visualization

2D visualization is perfect for planar robots and provides clear, interpretable plots for analysis and education.

### Basic 2D Robot Plotting

```python
import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin

def plot_2d_robot(robot, q, ax=None, **kwargs):
    """
    Plot a 2D robot configuration.
    
    Args:
        robot: Robot model
        q: Joint configuration
        ax: Matplotlib axis (optional)
        **kwargs: Additional plotting options
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Default plotting options
    options = {
        'link_color': 'blue',
        'joint_color': 'red',
        'link_width': 3,
        'joint_size': 8,
        'show_frames': True,
        'frame_scale': 0.1,
        'show_com': False,
        'com_color': 'green',
        'com_size': 6
    }
    options.update(kwargs)
    
    # Compute forward kinematics
    kinematic_state = pin.compute_forward_kinematics(robot, q)
    
    # Extract link positions
    link_positions = []
    link_orientations = []
    
    for link in robot.links:
        if link.name in kinematic_state.link_transforms:
            transform = kinematic_state.link_transforms[link.name]
            link_positions.append(transform.translation)
            link_orientations.append(transform.rotation)
    
    # Plot links
    for i in range(len(link_positions) - 1):
        start_pos = link_positions[i]
        end_pos = link_positions[i + 1]
        
        ax.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                color=options['link_color'], 
                linewidth=options['link_width'],
                solid_capstyle='round')
    
    # Plot joints
    for pos in link_positions:
        ax.plot(pos[0], pos[1], 'o', 
                color=options['joint_color'], 
                markersize=options['joint_size'])
    
    # Plot coordinate frames
    if options['show_frames']:
        for i, (pos, rot) in enumerate(zip(link_positions, link_orientations)):
            scale = options['frame_scale']
            
            # X-axis (red)
            x_end = pos + scale * rot[:, 0]
            ax.arrow(pos[0], pos[1], x_end[0] - pos[0], x_end[1] - pos[1],
                    head_width=scale*0.1, head_length=scale*0.05, 
                    fc='red', ec='red', alpha=0.7)
            
            # Y-axis (green)
            y_end = pos + scale * rot[:, 1]
            ax.arrow(pos[0], pos[1], y_end[0] - pos[0], y_end[1] - pos[1],
                    head_width=scale*0.1, head_length=scale*0.05, 
                    fc='green', ec='green', alpha=0.7)
    
    # Plot centers of mass
    if options['show_com']:
        for link in robot.links:
            if link.has_inertia and link.name in kinematic_state.link_transforms:
                transform = kinematic_state.link_transforms[link.name]
                com_world = pin.math.transform.transform_point(transform, link.center_of_mass)
                
                ax.plot(com_world[0], com_world[1], 's',
                       color=options['com_color'],
                       markersize=options['com_size'])
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Robot Configuration: {np.degrees(q)} degrees')
    
    return ax

# Example usage
def create_visualization_robot():
    """Create a robot for visualization examples."""
    links = []
    joints = []
    
    # Base
    base = pin.Link("base", mass=2.0)
    links.append(base)
    
    # Three links
    link_params = [
        {"mass": 1.5, "length": 0.4},
        {"mass": 1.0, "length": 0.3},
        {"mass": 0.5, "length": 0.2}
    ]
    
    for i, params in enumerate(link_params):
        link = pin.Link(
            name=f"link{i+1}",
            mass=params["mass"],
            center_of_mass=np.array([params["length"]/2, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, params["mass"]*params["length"]**2/12, params["mass"]*params["length"]**2/12])
        )
        links.append(link)
        
        parent = "base" if i == 0 else f"link{i}"
        offset = np.array([0, 0, 0.1]) if i == 0 else np.array([link_params[i-1]["length"], 0, 0])
        
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link=parent,
            child_link=f"link{i+1}",
            origin_transform=pin.create_transform(np.eye(3), offset)
        )
        joints.append(joint)
    
    return pin.create_robot_model("viz_robot", links, joints, "base")

# Create visualization examples
robot = create_visualization_robot()

# Multiple configurations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

configurations = [
    ("Home", np.array([0.0, 0.0, 0.0])),
    ("Reach Forward", np.array([0.0, -np.pi/4, -np.pi/2])),
    ("Reach Up", np.array([0.0, -np.pi/2, 0.0])),
    ("Folded", np.array([0.0, np.pi/2, np.pi/2]))
]

for i, (name, q) in enumerate(configurations):
    plot_2d_robot(robot, q, ax=axes[i], 
                  show_frames=True, 
                  show_com=True,
                  link_color='navy',
                  joint_color='red')
    axes[i].set_title(name)

plt.tight_layout()
plt.savefig('robot_configurations_2d.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Advanced 2D Visualization Features

```python
def plot_robot_with_workspace(robot, ax=None, workspace_samples=1000):
    """
    Plot robot with its reachable workspace.
    
    Args:
        robot: Robot model
        ax: Matplotlib axis
        workspace_samples: Number of random configurations to sample
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sample workspace
    workspace_points = []
    
    for _ in range(workspace_samples):
        # Random valid configuration
        q = []
        for joint_name in robot.actuated_joint_names:
            joint = robot.get_joint(joint_name)
            if joint:
                q_min = joint.position_limit_lower
                q_max = joint.position_limit_upper
                q.append(np.random.uniform(q_min, q_max))
        
        q = np.array(q)
        
        try:
            ee_pos = pin.get_link_position(robot, q, robot.links[-1].name)
            workspace_points.append(ee_pos[:2])  # Only X-Y for 2D
        except:
            continue
    
    workspace_points = np.array(workspace_points)
    
    # Plot workspace
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], 
              c='lightblue', alpha=0.3, s=1, label='Workspace')
    
    # Plot current robot configuration
    q_current = np.array([np.pi/6, np.pi/4, -np.pi/3])
    plot_2d_robot(robot, q_current, ax=ax, 
                  link_color='red', joint_color='darkred')
    
    # Highlight end-effector
    ee_pos = pin.get_link_position(robot, q_current, robot.links[-1].name)
    ax.plot(ee_pos[0], ee_pos[1], 'ro', markersize=10, label='End-effector')
    
    ax.legend()
    ax.set_title('Robot with Reachable Workspace')
    
    return ax

# Plot workspace
plot_robot_with_workspace(robot)
plt.savefig('robot_workspace_2d.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 3D Robot Rendering

For spatial robots and more realistic visualization, 3D rendering provides better understanding of complex geometries.

```python
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_robot(robot, q, ax=None, **kwargs):
    """
    Plot a 3D robot configuration.
    
    Args:
        robot: Robot model
        q: Joint configuration
        ax: 3D matplotlib axis
        **kwargs: Plotting options
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    options = {
        'link_color': 'blue',
        'joint_color': 'red',
        'link_width': 3,
        'joint_size': 8,
        'show_frames': True,
        'frame_scale': 0.1
    }
    options.update(kwargs)
    
    # Compute forward kinematics
    kinematic_state = pin.compute_forward_kinematics(robot, q)
    
    # Extract positions
    positions = []
    orientations = []
    
    for link in robot.links:
        if link.name in kinematic_state.link_transforms:
            transform = kinematic_state.link_transforms[link.name]
            positions.append(transform.translation)
            orientations.append(transform.rotation)
    
    positions = np.array(positions)
    
    # Plot links as lines
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        
        ax.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                color=options['link_color'], 
                linewidth=options['link_width'])
    
    # Plot joints as spheres
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c=options['joint_color'], s=options['joint_size']**2)
    
    # Plot coordinate frames
    if options['show_frames']:
        for pos, rot in zip(positions, orientations):
            scale = options['frame_scale']
            
            # X, Y, Z axes
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                end = pos + scale * rot[:, i]
                ax.plot([pos[0], end[0]], 
                       [pos[1], end[1]], 
                       [pos[2], end[2]], 
                       color=color, linewidth=2)
    
    # Formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Robot Configuration')
    
    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax

# Create 3D spatial robot
def create_3d_robot():
    """Create a 3D spatial robot."""
    links = []
    joints = []
    
    # Base
    base = pin.Link("base", mass=5.0)
    links.append(base)
    
    # 6-DOF arm
    link_params = [
        {"mass": 3.0, "offset": [0, 0, 0.3], "axis": [0, 0, 1]},  # Base rotation
        {"mass": 2.5, "offset": [0, 0, 0.1], "axis": [0, 1, 0]},  # Shoulder
        {"mass": 2.0, "offset": [0.4, 0, 0], "axis": [0, 1, 0]},  # Elbow
        {"mass": 1.5, "offset": [0.3, 0, 0], "axis": [1, 0, 0]},  # Wrist 1
        {"mass": 1.0, "offset": [0.1, 0, 0], "axis": [0, 1, 0]},  # Wrist 2
        {"mass": 0.5, "offset": [0.1, 0, 0], "axis": [0, 0, 1]},  # Wrist 3
    ]
    
    for i, params in enumerate(link_params):
        link = pin.Link(
            name=f"link{i+1}",
            mass=params["mass"],
            center_of_mass=np.array([0.05, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, 0.01, 0.01])
        )
        links.append(link)
        
        parent = "base" if i == 0 else f"link{i}"
        
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array(params["axis"]),
            parent_link=parent,
            child_link=f"link{i+1}",
            origin_transform=pin.create_transform(np.eye(3), np.array(params["offset"])),
            position_limit_lower=-np.pi,
            position_limit_upper=np.pi
        )
        joints.append(joint)
    
    return pin.create_robot_model("3d_robot", links, joints, "base")

# Visualize 3D robot
robot_3d = create_3d_robot()
q_3d = np.array([np.pi/4, -np.pi/6, np.pi/3, 0, np.pi/4, np.pi/6])

fig = plt.figure(figsize=(15, 12))

# Multiple views
for i, (elev, azim, title) in enumerate([(20, 45, 'Isometric'), 
                                        (0, 0, 'Front'), 
                                        (0, 90, 'Side'), 
                                        (90, 0, 'Top')]):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    plot_3d_robot(robot_3d, q_3d, ax=ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'{title} View')

plt.tight_layout()
plt.savefig('robot_3d_views.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Trajectory Visualization

Visualizing robot trajectories helps understand motion patterns and validate path planning algorithms.

```python
def plot_trajectory(robot, trajectory, dt=0.1, **kwargs):
    """
    Plot robot trajectory over time.
    
    Args:
        robot: Robot model
        trajectory: Array of joint configurations over time
        dt: Time step between configurations
        **kwargs: Plotting options
    """
    options = {
        'show_robot_poses': True,
        'pose_interval': 10,
        'trajectory_color': 'blue',
        'trajectory_width': 2,
        'pose_alpha': 0.3
    }
    options.update(kwargs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: End-effector trajectory
    ee_trajectory = []
    time_points = []
    
    for i, q in enumerate(trajectory):
        ee_pos = pin.get_link_position(robot, q, robot.links[-1].name)
        ee_trajectory.append(ee_pos)
        time_points.append(i * dt)
    
    ee_trajectory = np.array(ee_trajectory)
    
    # Plot end-effector path
    ax1.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], 
            color=options['trajectory_color'], 
            linewidth=options['trajectory_width'],
            label='End-effector path')
    
    # Plot robot poses at intervals
    if options['show_robot_poses']:
        for i in range(0, len(trajectory), options['pose_interval']):
            plot_2d_robot(robot, trajectory[i], ax=ax1,
                         link_color='gray',
                         joint_color='lightgray',
                         show_frames=False,
                         alpha=options['pose_alpha'])
    
    # Highlight start and end
    ax1.plot(ee_trajectory[0, 0], ee_trajectory[0, 1], 'go', 
            markersize=10, label='Start')
    ax1.plot(ee_trajectory[-1, 0], ee_trajectory[-1, 1], 'ro', 
            markersize=10, label='End')
    
    ax1.set_title('End-Effector Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: Joint angles over time
    time_points = np.array(time_points)
    
    for i in range(robot.num_dof):
        joint_angles = trajectory[:, i]
        ax2.plot(time_points, np.degrees(joint_angles), 
                label=f'Joint {i+1}', linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Joint Angle (degrees)')
    ax2.set_title('Joint Angles vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Generate example trajectory
def generate_circular_trajectory(robot, center, radius, num_points=50):
    """Generate circular end-effector trajectory."""
    trajectory = []
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        target_x = center[0] + radius * np.cos(angle)
        target_y = center[1] + radius * np.sin(angle)
        target_pos = np.array([target_x, target_y, center[2]])
        
        # Simple inverse kinematics (would use proper IK in practice)
        q_guess = np.random.uniform(-np.pi/2, np.pi/2, robot.num_dof)
        trajectory.append(q_guess)
    
    return np.array(trajectory)

# Plot trajectory
robot = create_visualization_robot()
traj = generate_circular_trajectory(robot, center=[0.4, 0.2, 0.1], radius=0.15)
plot_trajectory(robot, traj)
plt.savefig('robot_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
```

This visualization tutorial provides comprehensive tools for creating publication-quality robot visualizations, from basic 2D plots to advanced 3D renderings and trajectory analysis. The examples demonstrate practical applications for education, research, and engineering analysis.
