"""
Robot visualization utilities for py-pinocchio

This module provides simple visualization tools for educational purposes.
The focus is on clarity and understanding rather than advanced graphics.

Main components:
- RobotVisualizer: 2D/3D robot visualization
- plot_robot: Simple plotting functions
- animate_robot: Animation utilities
"""

from .robot_visualizer import RobotVisualizer, plot_robot_2d, plot_robot_3d
from .animation import animate_robot_motion, create_animation
from .plotting import plot_workspace, plot_jacobian_analysis

__all__ = [
    "RobotVisualizer",
    "plot_robot_2d",
    "plot_robot_3d", 
    "animate_robot_motion",
    "create_animation",
    "plot_workspace",
    "plot_jacobian_analysis",
]
