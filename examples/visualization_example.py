#!/usr/bin/env python3
"""
Robot visualization example for py-pinocchio

This example demonstrates the visualization capabilities of py-pinocchio,
including 2D/3D robot plotting, workspace analysis, and animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import py_pinocchio as pin

# Check if visualization is available
if not hasattr(pin, 'RobotVisualizer'):
    print("Visualization not available. Please install matplotlib:")
    print("pip install matplotlib")
    exit(1)


def create_2dof_planar_robot():
    """Create a 2-DOF planar robot for visualization."""
    print("Creating 2-DOF planar robot...")
    
    # Create links
    base = pin.Link("base", mass=0.0)
    link1 = pin.Link("link1", mass=1.0, center_of_mass=[0.5, 0, 0])
    link2 = pin.Link("link2", mass=0.5, center_of_mass=[0.4, 0, 0])
    
    # Create joints
    joint1 = pin.Joint(
        name="joint1",
        joint_type=pin.JointType.REVOLUTE,
        axis=[0, 0, 1],
        parent_link="base",
        child_link="link1",
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    
    joint2_origin = pin.create_transform(np.eye(3), [1.0, 0, 0])
    joint2 = pin.Joint(
        name="joint2",
        joint_type=pin.JointType.REVOLUTE,
        axis=[0, 0, 1],
        parent_link="link1",
        child_link="link2",
        origin_transform=joint2_origin,
        position_limit_lower=-np.pi,
        position_limit_upper=np.pi
    )
    
    robot = pin.create_robot_model("planar_2dof", [base, link1, link2], [joint1, joint2], "base")
    print(f"Created robot with {robot.num_dof} DOF")
    return robot


def demonstrate_basic_visualization(robot):
    """Demonstrate basic 2D and 3D robot visualization."""
    print("\n=== Basic Robot Visualization ===")
    
    # Test configurations
    configs = [
        np.array([0.0, 0.0]),           # Zero configuration
        np.array([np.pi/2, np.pi/4]),   # 90° and 45°
        np.array([np.pi/4, -np.pi/2]),  # 45° and -90°
        np.array([-np.pi/3, np.pi/3]),  # -60° and 60°
    ]
    
    config_names = ["Zero", "90°/45°", "45°/-90°", "-60°/60°"]
    
    # Create 2D visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (config, name) in enumerate(zip(configs, config_names)):
        pin.plot_robot_2d(robot, config, ax=axes[i])
        axes[i].set_title(f"Configuration: {name}")
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    plt.savefig("examples/robot_configurations_2d.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    
    for i, (config, name) in enumerate(zip(configs, config_names)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        pin.plot_robot_3d(robot, config, ax=ax)
        ax.set_title(f"Configuration: {name}")
    
    plt.tight_layout()
    plt.savefig("examples/robot_configurations_3d.png", dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_workspace_analysis(robot):
    """Demonstrate workspace visualization."""
    print("\n=== Workspace Analysis ===")
    
    # Plot workspace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reachable workspace
    pin.plot_workspace(robot, "link2", resolution=40, ax=ax1)
    ax1.set_title("Reachable Workspace")
    
    # Jacobian analysis
    pin.plot_jacobian_analysis(robot, "link2", resolution=30, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("examples/workspace_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_trajectory_visualization(robot):
    """Demonstrate trajectory plotting."""
    print("\n=== Trajectory Visualization ===")
    
    # Create a circular trajectory
    n_points = 50
    t = np.linspace(0, 2*np.pi, n_points)
    
    trajectory = []
    for time in t:
        # Simple circular motion in joint space
        q1 = 0.5 * np.sin(time)
        q2 = 0.3 * np.cos(2*time)
        trajectory.append(np.array([q1, q2]))
    
    # Plot trajectory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # End-effector trajectory in Cartesian space
    pin.plot_trajectory(robot, trajectory, "link2", ax=ax1)
    
    # Joint trajectories over time
    pin.plot_joint_trajectories(trajectory, time_steps=t.tolist(), 
                                joint_names=["Joint 1", "Joint 2"], ax=ax2)
    
    plt.tight_layout()
    plt.savefig("examples/trajectory_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_animation(robot):
    """Demonstrate robot animation."""
    print("\n=== Robot Animation ===")
    
    # Create pendulum-like motion
    n_frames = 60
    t = np.linspace(0, 4, n_frames)
    
    trajectory = []
    for time in t:
        # Pendulum motion for joint 1, oscillation for joint 2
        q1 = 0.8 * np.sin(2*np.pi * time / 4)
        q2 = 0.4 * np.cos(4*np.pi * time / 4)
        trajectory.append(np.array([q1, q2]))
    
    print("Creating animation... (close window to continue)")
    
    # Create animation
    anim = pin.animate_robot_motion(robot, trajectory, t.tolist(), mode='2d')
    
    # Show animation
    plt.show()
    
    print("Animation completed!")


def demonstrate_advanced_visualization(robot):
    """Demonstrate advanced visualization features."""
    print("\n=== Advanced Visualization ===")
    
    # Custom visualization configuration
    from py_pinocchio.visualization.robot_visualizer import VisualizationConfig
    
    config = VisualizationConfig(
        link_width=0.08,
        joint_size=0.15,
        link_color='darkblue',
        joint_color='orange',
        end_effector_color='red',
        show_frames=True,
        frame_scale=0.3
    )
    
    # Create visualizer with custom config
    visualizer = pin.RobotVisualizer(robot, config)
    
    # Plot with custom styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    config1 = np.array([np.pi/3, -np.pi/4])
    visualizer.plot_2d(config1, ax=ax1)
    ax1.set_title("Custom 2D Visualization")
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    
    # 3D with coordinate frames
    ax2 = fig.add_subplot(122, projection='3d')
    visualizer.plot_3d(config1, ax=ax2)
    ax2.set_title("3D with Coordinate Frames")
    
    plt.tight_layout()
    plt.savefig("examples/advanced_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()


def create_comparison_visualization():
    """Create comparison between different robot configurations."""
    print("\n=== Robot Comparison ===")
    
    # Create two different robots
    robot1 = create_2dof_planar_robot()
    
    # Create a different robot with different link lengths
    base2 = pin.Link("base", mass=0.0)
    link1_2 = pin.Link("link1", mass=1.0, center_of_mass=[0.3, 0, 0])
    link2_2 = pin.Link("link2", mass=0.5, center_of_mass=[0.6, 0, 0])
    
    joint1_2 = pin.Joint("joint1", pin.JointType.REVOLUTE, axis=[0, 0, 1], 
                        parent_link="base", child_link="link1")
    joint2_2_origin = pin.create_transform(np.eye(3), [0.6, 0, 0])
    joint2_2 = pin.Joint("joint2", pin.JointType.REVOLUTE, axis=[0, 0, 1],
                        parent_link="link1", child_link="link2", 
                        origin_transform=joint2_2_origin)
    
    robot2 = pin.create_robot_model("planar_2dof_v2", [base2, link1_2, link2_2], 
                                   [joint1_2, joint2_2], "base")
    
    # Compare workspaces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    pin.plot_workspace(robot1, "link2", resolution=30, ax=ax1)
    ax1.set_title("Robot 1 Workspace (L1=1.0, L2=0.8)")
    
    pin.plot_workspace(robot2, "link2", resolution=30, ax=ax2)
    ax2.set_title("Robot 2 Workspace (L1=0.6, L2=1.2)")
    
    plt.tight_layout()
    plt.savefig("examples/robot_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("py-pinocchio Visualization Example")
    print("=" * 50)
    
    # Create robot
    robot = create_2dof_planar_robot()
    
    # Run demonstrations
    demonstrate_basic_visualization(robot)
    demonstrate_workspace_analysis(robot)
    demonstrate_trajectory_visualization(robot)
    demonstrate_advanced_visualization(robot)
    create_comparison_visualization()
    
    # Animation (optional - can be slow)
    response = input("\nRun animation demo? (y/n): ")
    if response.lower() == 'y':
        demonstrate_animation(robot)
    
    print("\n" + "=" * 50)
    print("Visualization example completed!")
    print("\nGenerated files:")
    print("- robot_configurations_2d.png")
    print("- robot_configurations_3d.png") 
    print("- workspace_analysis.png")
    print("- trajectory_analysis.png")
    print("- advanced_visualization.png")
    print("- robot_comparison.png")
    
    print("\nVisualization features demonstrated:")
    print("- 2D and 3D robot plotting")
    print("- Workspace analysis and visualization")
    print("- Jacobian analysis (manipulability, singularities)")
    print("- Trajectory plotting and analysis")
    print("- Robot animation")
    print("- Custom visualization styling")
    print("- Robot comparison")


if __name__ == "__main__":
    main()
