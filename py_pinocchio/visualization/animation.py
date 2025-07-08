"""
Robot animation utilities

This module provides tools for animating robot motion, useful for
visualizing trajectories and understanding robot dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass

from ..model import RobotModel
from ..types import JointPositions
from .robot_visualizer import RobotVisualizer, VisualizationConfig


@dataclass
class AnimationConfig:
    """Configuration for robot animation."""
    fps: int = 30
    interval: int = 50  # milliseconds between frames
    repeat: bool = True
    save_path: Optional[str] = None
    dpi: int = 100
    bitrate: int = 1800


def animate_robot_motion(robot: RobotModel, 
                        trajectory: List[JointPositions],
                        time_steps: Optional[List[float]] = None,
                        mode: str = '2d',
                        vis_config: Optional[VisualizationConfig] = None,
                        anim_config: Optional[AnimationConfig] = None) -> animation.FuncAnimation:
    """
    Animate robot motion along a trajectory.
    
    Args:
        robot: Robot model to animate
        trajectory: List of joint configurations
        time_steps: Time stamps for each configuration (optional)
        mode: '2d' or '3d' visualization
        vis_config: Visualization configuration
        anim_config: Animation configuration
        
    Returns:
        Matplotlib animation object
    """
    if not trajectory:
        raise ValueError("Trajectory cannot be empty")
    
    vis_config = vis_config or VisualizationConfig()
    anim_config = anim_config or AnimationConfig()
    
    # Create visualizer
    visualizer = RobotVisualizer(robot, vis_config)
    
    # Set up figure and axes
    if mode == '3d':
        fig = plt.figure(figsize=vis_config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=vis_config.figure_size)
    
    # Animation function
    def animate_frame(frame_idx: int):
        """Update function for animation."""
        ax.clear()
        
        # Get current configuration
        current_config = trajectory[frame_idx]
        
        # Plot robot at current configuration
        if mode == '3d':
            visualizer.plot_3d(current_config, ax=ax, show_labels=False)
        else:
            visualizer.plot_2d(current_config, ax=ax, show_labels=False)
        
        # Add frame information
        if time_steps:
            time_text = f"Time: {time_steps[frame_idx]:.2f}s"
        else:
            time_text = f"Frame: {frame_idx}/{len(trajectory)-1}"
        
        ax.set_title(f"{robot.name} - {time_text}")
        
        return ax.get_children()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(trajectory),
        interval=anim_config.interval, repeat=anim_config.repeat,
        blit=False
    )
    
    # Save animation if requested
    if anim_config.save_path:
        anim.save(anim_config.save_path, writer='ffmpeg', 
                 fps=anim_config.fps, dpi=anim_config.dpi,
                 bitrate=anim_config.bitrate)
    
    return anim


def create_animation(robot: RobotModel,
                    trajectory_func: Callable[[float], JointPositions],
                    duration: float,
                    fps: int = 30,
                    mode: str = '2d',
                    vis_config: Optional[VisualizationConfig] = None) -> animation.FuncAnimation:
    """
    Create animation from a trajectory function.
    
    Args:
        robot: Robot model to animate
        trajectory_func: Function that takes time and returns joint positions
        duration: Total animation duration in seconds
        fps: Frames per second
        mode: '2d' or '3d' visualization
        vis_config: Visualization configuration
        
    Returns:
        Matplotlib animation object
    """
    # Generate trajectory
    time_steps = np.linspace(0, duration, int(duration * fps))
    trajectory = [trajectory_func(t) for t in time_steps]
    
    # Create animation config
    anim_config = AnimationConfig(fps=fps, interval=int(1000/fps))
    
    return animate_robot_motion(robot, trajectory, time_steps.tolist(), 
                               mode, vis_config, anim_config)


def animate_pendulum_motion(robot: RobotModel, 
                           initial_angle: float = np.pi/3,
                           duration: float = 3.0,
                           fps: int = 30) -> animation.FuncAnimation:
    """
    Animate simple pendulum motion (for 1-DOF robots).
    
    Args:
        robot: 1-DOF robot model
        initial_angle: Initial pendulum angle
        duration: Animation duration
        fps: Frames per second
        
    Returns:
        Animation of pendulum motion
    """
    if robot.num_dof != 1:
        raise ValueError("Pendulum animation requires 1-DOF robot")
    
    # Simple pendulum equation: θ(t) = θ₀ * cos(√(g/l) * t)
    # Assuming unit length and gravity
    omega = np.sqrt(9.81)  # Natural frequency
    
    def pendulum_trajectory(t: float) -> JointPositions:
        angle = initial_angle * np.cos(omega * t)
        return np.array([angle])
    
    return create_animation(robot, pendulum_trajectory, duration, fps, mode='2d')


def animate_circular_motion(robot: RobotModel,
                           radius: float = 0.5,
                           center: Tuple[float, float] = (0.5, 0.0),
                           period: float = 4.0,
                           fps: int = 30) -> animation.FuncAnimation:
    """
    Animate robot end-effector moving in a circle (for 2-DOF planar robots).
    
    Args:
        robot: 2-DOF planar robot model
        radius: Circle radius
        center: Circle center (x, y)
        period: Time for one complete circle
        fps: Frames per second
        
    Returns:
        Animation of circular motion
    """
    if robot.num_dof != 2:
        raise ValueError("Circular motion animation requires 2-DOF robot")
    
    def circular_trajectory(t: float) -> JointPositions:
        # Parametric circle
        angle = 2 * np.pi * t / period
        target_x = center[0] + radius * np.cos(angle)
        target_y = center[1] + radius * np.sin(angle)
        
        # Simple inverse kinematics for 2-DOF planar robot
        # Assuming link lengths of 1.0 each
        l1 = l2 = 1.0
        
        # Distance from origin to target
        d = np.sqrt(target_x**2 + target_y**2)
        
        # Check if target is reachable
        if d > l1 + l2:
            d = l1 + l2 - 0.01  # Slightly inside workspace
            target_x = target_x * d / np.sqrt(target_x**2 + target_y**2)
            target_y = target_y * d / np.sqrt(target_x**2 + target_y**2)
        
        # Inverse kinematics
        cos_q2 = (target_x**2 + target_y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1, 1)
        
        q2 = np.arccos(cos_q2)
        q1 = np.arctan2(target_y, target_x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        
        return np.array([q1, q2])
    
    return create_animation(robot, circular_trajectory, period, fps, mode='2d')


def save_animation_frames(robot: RobotModel,
                         trajectory: List[JointPositions],
                         output_dir: str,
                         mode: str = '2d',
                         vis_config: Optional[VisualizationConfig] = None) -> None:
    """
    Save individual frames of robot animation as images.
    
    Args:
        robot: Robot model
        trajectory: Joint trajectory
        output_dir: Directory to save frames
        mode: '2d' or '3d' visualization
        vis_config: Visualization configuration
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    vis_config = vis_config or VisualizationConfig()
    visualizer = RobotVisualizer(robot, vis_config)
    
    for i, config in enumerate(trajectory):
        # Create figure
        if mode == '3d':
            fig = plt.figure(figsize=vis_config.figure_size)
            ax = fig.add_subplot(111, projection='3d')
            visualizer.plot_3d(config, ax=ax, show_labels=False)
        else:
            fig, ax = plt.subplots(figsize=vis_config.figure_size)
            visualizer.plot_2d(config, ax=ax, show_labels=False)
        
        ax.set_title(f"{robot.name} - Frame {i:04d}")
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(trajectory)} frames to {output_dir}")


def create_comparison_animation(robots: List[RobotModel],
                               trajectories: List[List[JointPositions]],
                               labels: List[str],
                               mode: str = '2d') -> animation.FuncAnimation:
    """
    Create side-by-side animation comparing multiple robots.
    
    Args:
        robots: List of robot models
        trajectories: List of trajectories for each robot
        labels: Labels for each robot
        mode: '2d' or '3d' visualization
        
    Returns:
        Comparison animation
    """
    if len(robots) != len(trajectories) or len(robots) != len(labels):
        raise ValueError("Number of robots, trajectories, and labels must match")
    
    n_robots = len(robots)
    
    # Create subplots
    if mode == '3d':
        fig = plt.figure(figsize=(5*n_robots, 5))
        axes = [fig.add_subplot(1, n_robots, i+1, projection='3d') for i in range(n_robots)]
    else:
        fig, axes = plt.subplots(1, n_robots, figsize=(5*n_robots, 5))
        if n_robots == 1:
            axes = [axes]
    
    # Create visualizers
    visualizers = [RobotVisualizer(robot) for robot in robots]
    
    def animate_frame(frame_idx: int):
        """Update function for comparison animation."""
        for i, (visualizer, trajectory, label, ax) in enumerate(zip(visualizers, trajectories, labels, axes)):
            ax.clear()
            
            if frame_idx < len(trajectory):
                config = trajectory[frame_idx]
                if mode == '3d':
                    visualizer.plot_3d(config, ax=ax, show_labels=False)
                else:
                    visualizer.plot_2d(config, ax=ax, show_labels=False)
            
            ax.set_title(f"{label} - Frame {frame_idx}")
        
        return [child for ax in axes for child in ax.get_children()]
    
    # Find maximum trajectory length
    max_frames = max(len(traj) for traj in trajectories)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=max_frames,
        interval=50, repeat=True, blit=False
    )
    
    return anim
