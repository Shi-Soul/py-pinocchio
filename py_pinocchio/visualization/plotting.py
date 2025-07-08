"""
Plotting utilities for robot analysis

This module provides plotting functions for workspace analysis,
Jacobian visualization, and other robot analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable
from mpl_toolkits.mplot3d import Axes3D

from ..model import RobotModel
from ..algorithms.kinematics import get_link_position
from ..algorithms.jacobian import compute_geometric_jacobian, analyze_jacobian_singularities
from ..types import JointPositions


def plot_workspace(robot: RobotModel, 
                  end_effector_link: str,
                  joint_limits: Optional[List[Tuple[float, float]]] = None,
                  resolution: int = 50,
                  mode: str = '2d',
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot robot workspace by sampling joint configurations.
    
    Args:
        robot: Robot model
        end_effector_link: Name of end-effector link
        joint_limits: Custom joint limits [(min, max), ...] for each DOF
        resolution: Number of samples per joint
        mode: '2d' or '3d' plotting
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes with workspace plot
    """
    if robot.num_dof > 3:
        print(f"Warning: Workspace plotting for {robot.num_dof}-DOF robot may be slow")
    
    # Use robot joint limits if not provided
    if joint_limits is None:
        joint_limits = []
        actuated_joints = [robot.get_joint(name) for name in robot.actuated_joint_names]
        for joint in actuated_joints:
            if joint:
                joint_limits.append((joint.position_limit_lower, joint.position_limit_upper))
            else:
                joint_limits.append((-np.pi, np.pi))
    
    # Sample workspace
    workspace_points = []
    
    if robot.num_dof == 1:
        # 1-DOF: Simple line sampling
        q_samples = np.linspace(joint_limits[0][0], joint_limits[0][1], resolution)
        for q in q_samples:
            try:
                pos = get_link_position(robot, np.array([q]), end_effector_link)
                workspace_points.append(pos)
            except:
                continue
                
    elif robot.num_dof == 2:
        # 2-DOF: Grid sampling
        q1_samples = np.linspace(joint_limits[0][0], joint_limits[0][1], resolution)
        q2_samples = np.linspace(joint_limits[1][0], joint_limits[1][1], resolution)
        
        for q1 in q1_samples:
            for q2 in q2_samples:
                try:
                    pos = get_link_position(robot, np.array([q1, q2]), end_effector_link)
                    workspace_points.append(pos)
                except:
                    continue
                    
    elif robot.num_dof == 3:
        # 3-DOF: Reduced resolution grid sampling
        reduced_res = max(10, resolution // 3)
        q1_samples = np.linspace(joint_limits[0][0], joint_limits[0][1], reduced_res)
        q2_samples = np.linspace(joint_limits[1][0], joint_limits[1][1], reduced_res)
        q3_samples = np.linspace(joint_limits[2][0], joint_limits[2][1], reduced_res)
        
        for q1 in q1_samples:
            for q2 in q2_samples:
                for q3 in q3_samples:
                    try:
                        pos = get_link_position(robot, np.array([q1, q2, q3]), end_effector_link)
                        workspace_points.append(pos)
                    except:
                        continue
    else:
        # Higher DOF: Random sampling
        n_samples = resolution ** 2
        for _ in range(n_samples):
            q = []
            for (q_min, q_max) in joint_limits:
                q.append(np.random.uniform(q_min, q_max))
            
            try:
                pos = get_link_position(robot, np.array(q), end_effector_link)
                workspace_points.append(pos)
            except:
                continue
    
    if not workspace_points:
        raise ValueError("No valid workspace points found")
    
    workspace_points = np.array(workspace_points)
    
    # Create plot
    if mode == '3d' or workspace_points.shape[1] == 3:
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2],
                  c='blue', alpha=0.6, s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Workspace: {robot.name} - {end_effector_link}')
        
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(workspace_points[:, 0], workspace_points[:, 1], 
                  c='blue', alpha=0.6, s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Workspace: {robot.name} - {end_effector_link}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    return ax


def plot_jacobian_analysis(robot: RobotModel,
                          end_effector_link: str,
                          joint_limits: Optional[List[Tuple[float, float]]] = None,
                          resolution: int = 30,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot Jacobian analysis (manipulability, condition number, singularities).
    
    Args:
        robot: Robot model
        end_effector_link: Name of end-effector link
        joint_limits: Custom joint limits
        resolution: Number of samples per joint
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes with Jacobian analysis
    """
    if robot.num_dof != 2:
        raise ValueError("Jacobian analysis plotting currently supports only 2-DOF robots")
    
    # Use robot joint limits if not provided
    if joint_limits is None:
        actuated_joints = [robot.get_joint(name) for name in robot.actuated_joint_names]
        joint_limits = []
        for joint in actuated_joints:
            if joint:
                joint_limits.append((joint.position_limit_lower, joint.position_limit_upper))
            else:
                joint_limits.append((-np.pi, np.pi))
    
    # Sample configuration space
    q1_samples = np.linspace(joint_limits[0][0], joint_limits[0][1], resolution)
    q2_samples = np.linspace(joint_limits[1][0], joint_limits[1][1], resolution)
    
    Q1, Q2 = np.meshgrid(q1_samples, q2_samples)
    manipulability = np.zeros_like(Q1)
    condition_numbers = np.zeros_like(Q1)
    
    for i in range(resolution):
        for j in range(resolution):
            q = np.array([Q1[i, j], Q2[i, j]])
            
            try:
                jacobian = compute_geometric_jacobian(robot, q, end_effector_link)
                analysis = analyze_jacobian_singularities(jacobian)
                
                manipulability[i, j] = analysis['manipulability']
                condition_numbers[i, j] = analysis['condition_number']
                
                # Cap condition number for visualization
                if condition_numbers[i, j] > 1000:
                    condition_numbers[i, j] = 1000
                    
            except:
                manipulability[i, j] = 0
                condition_numbers[i, j] = 1000
    
    # Create plot
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        # If single axis provided, create manipulability plot only
        ax1 = ax
        ax2 = None
    
    # Manipulability plot
    im1 = ax1.contourf(Q1, Q2, manipulability, levels=20, cmap='viridis')
    ax1.set_xlabel('Joint 1 (rad)')
    ax1.set_ylabel('Joint 2 (rad)')
    ax1.set_title('Manipulability')
    plt.colorbar(im1, ax=ax1)
    
    # Add singularity contour
    ax1.contour(Q1, Q2, manipulability, levels=[0.01], colors='red', linewidths=2)
    
    # Condition number plot (if second axis available)
    if ax2 is not None:
        im2 = ax2.contourf(Q1, Q2, condition_numbers, levels=20, cmap='plasma')
        ax2.set_xlabel('Joint 1 (rad)')
        ax2.set_ylabel('Joint 2 (rad)')
        ax2.set_title('Condition Number')
        plt.colorbar(im2, ax=ax2)
        
        # Add high condition number contour
        ax2.contour(Q1, Q2, condition_numbers, levels=[100], colors='red', linewidths=2)
    
    return ax1 if ax2 is None else (ax1, ax2)


def plot_trajectory(robot: RobotModel,
                   trajectory: List[JointPositions],
                   end_effector_link: str,
                   time_steps: Optional[List[float]] = None,
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot end-effector trajectory in Cartesian space.
    
    Args:
        robot: Robot model
        trajectory: List of joint configurations
        end_effector_link: Name of end-effector link
        time_steps: Time stamps for each configuration
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes with trajectory plot
    """
    # Compute end-effector positions
    ee_positions = []
    for config in trajectory:
        try:
            pos = get_link_position(robot, config, end_effector_link)
            ee_positions.append(pos)
        except:
            continue
    
    if not ee_positions:
        raise ValueError("No valid trajectory points found")
    
    ee_positions = np.array(ee_positions)
    
    # Create plot
    if ee_positions.shape[1] == 3:
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
               'b-', linewidth=2, label='Trajectory')
        
        # Mark start and end
        ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
                  c='green', s=100, label='Start')
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                  c='red', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectory
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        # Mark start and end
        ax.scatter(ee_positions[0, 0], ee_positions[0, 1], c='green', s=100, label='Start')
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], c='red', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'End-Effector Trajectory: {end_effector_link}')
    ax.legend()
    
    return ax


def plot_joint_trajectories(trajectory: List[JointPositions],
                           time_steps: Optional[List[float]] = None,
                           joint_names: Optional[List[str]] = None,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot joint trajectories over time.
    
    Args:
        trajectory: List of joint configurations
        time_steps: Time stamps for each configuration
        joint_names: Names of joints for legend
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes with joint trajectory plots
    """
    if not trajectory:
        raise ValueError("Trajectory cannot be empty")
    
    trajectory_array = np.array(trajectory)
    n_joints = trajectory_array.shape[1]
    
    if time_steps is None:
        time_steps = list(range(len(trajectory)))
    
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(n_joints)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each joint trajectory
    for i in range(n_joints):
        ax.plot(time_steps, trajectory_array[:, i], 
               label=joint_names[i], linewidth=2)
    
    ax.set_xlabel('Time (s)' if isinstance(time_steps[0], float) else 'Step')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Joint Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
