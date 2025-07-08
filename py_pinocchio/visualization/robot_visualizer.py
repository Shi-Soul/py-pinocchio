"""
Robot visualization using matplotlib

This module provides simple 2D and 3D robot visualization for educational purposes.
The visualizations help understand robot kinematics and dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from ..model import RobotModel, JointType
from ..algorithms.kinematics import compute_forward_kinematics
from ..types import JointPositions, Vector3


@dataclass
class VisualizationConfig:
    """Configuration for robot visualization."""
    link_width: float = 0.05
    joint_size: float = 0.1
    link_color: str = 'blue'
    joint_color: str = 'red'
    base_color: str = 'black'
    end_effector_color: str = 'green'
    show_frames: bool = True
    frame_scale: float = 0.2
    figure_size: Tuple[float, float] = (10, 8)


class RobotVisualizer:
    """
    Simple robot visualizer for educational purposes.
    
    Provides 2D and 3D visualization of robot configurations,
    workspace analysis, and motion animation.
    """
    
    def __init__(self, robot: RobotModel, config: Optional[VisualizationConfig] = None):
        """
        Initialize robot visualizer.
        
        Args:
            robot: Robot model to visualize
            config: Visualization configuration
        """
        self.robot = robot
        self.config = config or VisualizationConfig()
        
    def plot_2d(self, joint_positions: JointPositions, 
                ax: Optional[plt.Axes] = None, 
                show_labels: bool = True) -> plt.Axes:
        """
        Plot robot in 2D (XY plane).
        
        Args:
            joint_positions: Joint configuration to visualize
            ax: Matplotlib axes (creates new if None)
            show_labels: Whether to show link/joint labels
            
        Returns:
            Matplotlib axes with robot plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Compute forward kinematics
        kinematic_state = compute_forward_kinematics(self.robot, joint_positions)
        
        # Extract link positions
        link_positions = {}
        for link_name, transform in kinematic_state.link_transforms.items():
            link_positions[link_name] = transform.translation
        
        # Plot links as lines
        self._plot_links_2d(ax, link_positions, show_labels)
        
        # Plot joints as circles
        self._plot_joints_2d(ax, link_positions, show_labels)
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Robot Configuration: {self.robot.name}')
        
        return ax
    
    def plot_3d(self, joint_positions: JointPositions,
                ax: Optional[Axes3D] = None,
                show_labels: bool = True) -> Axes3D:
        """
        Plot robot in 3D.
        
        Args:
            joint_positions: Joint configuration to visualize
            ax: 3D matplotlib axes (creates new if None)
            show_labels: Whether to show link/joint labels
            
        Returns:
            3D matplotlib axes with robot plot
        """
        if ax is None:
            fig = plt.figure(figsize=self.config.figure_size)
            ax = fig.add_subplot(111, projection='3d')
        
        # Compute forward kinematics
        kinematic_state = compute_forward_kinematics(self.robot, joint_positions)
        
        # Extract link positions
        link_positions = {}
        link_orientations = {}
        for link_name, transform in kinematic_state.link_transforms.items():
            link_positions[link_name] = transform.translation
            link_orientations[link_name] = transform.rotation
        
        # Plot links as lines
        self._plot_links_3d(ax, link_positions, show_labels)
        
        # Plot joints as spheres
        self._plot_joints_3d(ax, link_positions, show_labels)
        
        # Plot coordinate frames if requested
        if self.config.show_frames:
            self._plot_frames_3d(ax, link_positions, link_orientations)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Robot Configuration: {self.robot.name}')
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d(ax, link_positions)
        
        return ax
    
    def _plot_links_2d(self, ax: plt.Axes, link_positions: Dict[str, Vector3], 
                      show_labels: bool) -> None:
        """Plot robot links in 2D."""
        # Build kinematic chain
        for joint in self.robot.joints:
            if joint.parent_link and joint.child_link:
                parent_pos = link_positions.get(joint.parent_link)
                child_pos = link_positions.get(joint.child_link)
                
                if parent_pos is not None and child_pos is not None:
                    # Plot link as line
                    ax.plot([parent_pos[0], child_pos[0]], 
                           [parent_pos[1], child_pos[1]], 
                           color=self.config.link_color, 
                           linewidth=self.config.link_width * 100,
                           solid_capstyle='round')
                    
                    # Add link label
                    if show_labels:
                        mid_x = (parent_pos[0] + child_pos[0]) / 2
                        mid_y = (parent_pos[1] + child_pos[1]) / 2
                        ax.text(mid_x, mid_y, joint.child_link, 
                               fontsize=8, ha='center', va='bottom')
    
    def _plot_joints_2d(self, ax: plt.Axes, link_positions: Dict[str, Vector3],
                       show_labels: bool) -> None:
        """Plot robot joints in 2D."""
        for link_name, position in link_positions.items():
            # Different colors for different link types
            if link_name == self.robot.root_link:
                color = self.config.base_color
                size = self.config.joint_size * 1.5
            elif self._is_end_effector(link_name):
                color = self.config.end_effector_color
                size = self.config.joint_size * 1.2
            else:
                color = self.config.joint_color
                size = self.config.joint_size
            
            # Plot joint as circle
            circle = plt.Circle((position[0], position[1]), size, 
                              color=color, zorder=10)
            ax.add_patch(circle)
            
            # Add joint label
            if show_labels:
                ax.text(position[0], position[1] + size * 1.5, link_name,
                       fontsize=8, ha='center', va='bottom')
    
    def _plot_links_3d(self, ax: Axes3D, link_positions: Dict[str, Vector3],
                      show_labels: bool) -> None:
        """Plot robot links in 3D."""
        for joint in self.robot.joints:
            if joint.parent_link and joint.child_link:
                parent_pos = link_positions.get(joint.parent_link)
                child_pos = link_positions.get(joint.child_link)
                
                if parent_pos is not None and child_pos is not None:
                    # Plot link as line
                    ax.plot([parent_pos[0], child_pos[0]], 
                           [parent_pos[1], child_pos[1]], 
                           [parent_pos[2], child_pos[2]],
                           color=self.config.link_color, 
                           linewidth=self.config.link_width * 100)
                    
                    # Add link label
                    if show_labels:
                        mid_pos = (parent_pos + child_pos) / 2
                        ax.text(mid_pos[0], mid_pos[1], mid_pos[2], 
                               joint.child_link, fontsize=8)
    
    def _plot_joints_3d(self, ax: Axes3D, link_positions: Dict[str, Vector3],
                       show_labels: bool) -> None:
        """Plot robot joints in 3D."""
        for link_name, position in link_positions.items():
            # Different colors for different link types
            if link_name == self.robot.root_link:
                color = self.config.base_color
                size = self.config.joint_size * 1.5
            elif self._is_end_effector(link_name):
                color = self.config.end_effector_color
                size = self.config.joint_size * 1.2
            else:
                color = self.config.joint_color
                size = self.config.joint_size
            
            # Plot joint as sphere (approximated with scatter)
            ax.scatter(position[0], position[1], position[2], 
                      c=color, s=size*1000, alpha=0.8)
            
            # Add joint label
            if show_labels:
                ax.text(position[0], position[1], position[2] + size, 
                       link_name, fontsize=8)
    
    def _plot_frames_3d(self, ax: Axes3D, link_positions: Dict[str, Vector3],
                       link_orientations: Dict[str, np.ndarray]) -> None:
        """Plot coordinate frames for each link."""
        scale = self.config.frame_scale
        
        for link_name, position in link_positions.items():
            if link_name in link_orientations:
                R = link_orientations[link_name]
                
                # X-axis (red)
                x_end = position + scale * R[:, 0]
                ax.plot([position[0], x_end[0]], 
                       [position[1], x_end[1]], 
                       [position[2], x_end[2]], 'r-', linewidth=2)
                
                # Y-axis (green)
                y_end = position + scale * R[:, 1]
                ax.plot([position[0], y_end[0]], 
                       [position[1], y_end[1]], 
                       [position[2], y_end[2]], 'g-', linewidth=2)
                
                # Z-axis (blue)
                z_end = position + scale * R[:, 2]
                ax.plot([position[0], z_end[0]], 
                       [position[1], z_end[1]], 
                       [position[2], z_end[2]], 'b-', linewidth=2)
    
    def _is_end_effector(self, link_name: str) -> bool:
        """Check if link is an end effector (has no children)."""
        return len(self.robot.get_child_links(link_name)) == 0
    
    def _set_equal_aspect_3d(self, ax: Axes3D, link_positions: Dict[str, Vector3]) -> None:
        """Set equal aspect ratio for 3D plot."""
        if not link_positions:
            return
        
        positions = np.array(list(link_positions.values()))
        
        # Get the range of each axis
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()
        
        # Use the maximum range for all axes
        max_range = max(x_range, y_range, z_range)
        if max_range == 0:
            max_range = 1.0
        
        # Center the plot
        x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
        y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
        z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)


def plot_robot_2d(robot: RobotModel, joint_positions: JointPositions,
                  config: Optional[VisualizationConfig] = None,
                  show_labels: bool = True,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Convenience function to plot robot in 2D.

    Args:
        robot: Robot model to visualize
        joint_positions: Joint configuration
        config: Visualization configuration
        show_labels: Whether to show labels
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes with robot plot
    """
    visualizer = RobotVisualizer(robot, config)
    return visualizer.plot_2d(joint_positions, ax=ax, show_labels=show_labels)


def plot_robot_3d(robot: RobotModel, joint_positions: JointPositions,
                  config: Optional[VisualizationConfig] = None,
                  show_labels: bool = True,
                  ax: Optional[Axes3D] = None) -> Axes3D:
    """
    Convenience function to plot robot in 3D.

    Args:
        robot: Robot model to visualize
        joint_positions: Joint configuration
        config: Visualization configuration
        show_labels: Whether to show labels
        ax: 3D matplotlib axes (creates new if None)

    Returns:
        3D matplotlib axes with robot plot
    """
    visualizer = RobotVisualizer(robot, config)
    return visualizer.plot_3d(joint_positions, ax=ax, show_labels=show_labels)
