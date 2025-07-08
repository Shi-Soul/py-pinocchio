"""
Forward kinematics algorithms using functional programming style

This module implements forward kinematics - computing the positions and orientations
of all robot links given joint positions. The algorithms are implemented as pure
functions for educational clarity.

Forward kinematics is fundamental to robotics as it answers: "Where are all the
robot links when the joints are at specific positions?"
"""

import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

from ..model import RobotModel, Joint, JointType
from ..math.transform import (
    Transform, 
    create_identity_transform,
    create_transform,
    compose_transforms,
    transform_point
)
from ..math.rotation import rodrigues_formula


class KinematicState(NamedTuple):
    """
    Immutable kinematic state containing all link transforms.
    
    This represents the complete kinematic configuration of the robot
    at a specific joint configuration.
    """
    joint_positions: np.ndarray
    link_transforms: Dict[str, Transform]  # link_name -> transform from world
    joint_transforms: Dict[str, Transform]  # joint_name -> transform from parent


def compute_forward_kinematics(robot: RobotModel, joint_positions: np.ndarray) -> KinematicState:
    """
    Compute forward kinematics for entire robot.
    
    This is the main forward kinematics function that computes the position
    and orientation of every link in the robot given joint positions.
    
    Algorithm:
    1. Start from root link (world frame)
    2. Traverse kinematic tree in topological order
    3. For each joint, compute transformation based on joint type and position
    4. Accumulate transformations to get world frame poses
    
    Args:
        robot: Robot model
        joint_positions: Joint positions for actuated joints
        
    Returns:
        KinematicState with all link and joint transforms
        
    Raises:
        ValueError: If joint_positions has wrong size or values out of limits
    """
    # Validate input
    if len(joint_positions) != robot.num_dof:
        raise ValueError(f"Expected {robot.num_dof} joint positions, got {len(joint_positions)}")
    
    if not robot.is_valid_joint_configuration(joint_positions):
        raise ValueError("Joint positions exceed joint limits")
    
    # Create joint position mapping
    actuated_joints = robot.actuated_joint_names
    joint_pos_map = {name: joint_positions[i] for i, name in enumerate(actuated_joints)}
    
    # Initialize transforms
    link_transforms = {}
    joint_transforms = {}
    
    # Root link starts at identity (world frame)
    if robot.root_link:
        link_transforms[robot.root_link] = create_identity_transform()
    
    # Traverse kinematic tree
    _traverse_kinematic_tree(
        robot=robot,
        current_link=robot.root_link,
        joint_pos_map=joint_pos_map,
        link_transforms=link_transforms,
        joint_transforms=joint_transforms
    )
    
    return KinematicState(
        joint_positions=joint_positions.copy(),
        link_transforms=link_transforms,
        joint_transforms=joint_transforms
    )


def _traverse_kinematic_tree(robot: RobotModel, 
                           current_link: str,
                           joint_pos_map: Dict[str, float],
                           link_transforms: Dict[str, Transform],
                           joint_transforms: Dict[str, Transform]) -> None:
    """
    Recursively traverse kinematic tree and compute transforms.
    
    This is a depth-first traversal that computes transforms for all
    links reachable from the current link.
    """
    if not current_link:
        return
    
    current_transform = link_transforms.get(current_link, create_identity_transform())
    
    # Process all child links
    for child_link in robot.get_child_links(current_link):
        joint = robot.get_joint_between_links(current_link, child_link)
        if not joint:
            continue
        
        # Compute joint transformation in local frame
        joint_transform_local = _compute_joint_transform(joint, joint_pos_map)

        # Compute joint transform in world frame: T_joint_world = T_parent * T_joint_local
        joint_transform_world = compose_transforms(current_transform, joint_transform_local)
        joint_transforms[joint.name] = joint_transform_world

        # Child link transform is the same as joint transform in world frame
        child_transform = joint_transform_world
        link_transforms[child_link] = child_transform
        
        # Recursively process child's children
        _traverse_kinematic_tree(
            robot=robot,
            current_link=child_link,
            joint_pos_map=joint_pos_map,
            link_transforms=link_transforms,
            joint_transforms=joint_transforms
        )


def _compute_joint_transform(joint: Joint, joint_pos_map: Dict[str, float]) -> Transform:
    """
    Compute transformation for a single joint.
    
    This function implements the kinematic equation for each joint type:
    - Fixed: No additional transformation
    - Revolute: Rotation about joint axis
    - Prismatic: Translation along joint axis
    
    Args:
        joint: Joint to compute transform for
        joint_pos_map: Mapping of joint names to positions
        
    Returns:
        Transform representing joint motion
    """
    # Start with joint's origin transform (from URDF)
    base_transform = joint.origin_transform
    
    if joint.joint_type == JointType.FIXED:
        # Fixed joint: only origin transform
        return base_transform
    
    elif joint.joint_type == JointType.REVOLUTE:
        # Revolute joint: rotation about axis at joint location
        joint_pos = joint_pos_map.get(joint.name, 0.0)

        # Create rotation matrix using Rodrigues' formula
        rotation_matrix = rodrigues_formula(joint.axis, joint_pos)

        # For revolute joints, we need to:
        # 1. Apply origin transform to get to joint location
        # 2. Apply rotation at that location
        # This gives us: T_total = T_origin * T_rotation
        motion_transform = create_transform(rotation_matrix, np.zeros(3))
        return compose_transforms(base_transform, motion_transform)
    
    elif joint.joint_type == JointType.PRISMATIC:
        # Prismatic joint: translation along axis
        joint_pos = joint_pos_map.get(joint.name, 0.0)
        
        # Create translation
        translation = joint_pos * joint.axis
        motion_transform = create_transform(np.eye(3), translation)
        
        # Combine: T_total = T_origin * T_motion
        return compose_transforms(base_transform, motion_transform)
    
    else:
        # Other joint types not implemented yet
        raise NotImplementedError(f"Joint type {joint.joint_type} not implemented")


def compute_link_transforms(robot: RobotModel, joint_positions: np.ndarray) -> Dict[str, Transform]:
    """
    Compute transforms for all links (convenience function).
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        
    Returns:
        Dictionary mapping link names to world frame transforms
    """
    kinematic_state = compute_forward_kinematics(robot, joint_positions)
    return kinematic_state.link_transforms


def get_link_transform(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> Transform:
    """
    Get transform for a specific link.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        link_name: Name of link to get transform for
        
    Returns:
        Transform from world frame to link frame
        
    Raises:
        ValueError: If link doesn't exist
    """
    if not robot.get_link(link_name):
        raise ValueError(f"Link '{link_name}' not found in robot model")
    
    link_transforms = compute_link_transforms(robot, joint_positions)
    return link_transforms.get(link_name, create_identity_transform())


def get_link_position(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray:
    """
    Get position of a specific link's origin.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        link_name: Name of link
        
    Returns:
        3D position vector in world frame
    """
    transform = get_link_transform(robot, joint_positions, link_name)
    return transform.translation


def get_link_orientation(robot: RobotModel, joint_positions: np.ndarray, link_name: str) -> np.ndarray:
    """
    Get orientation of a specific link.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        link_name: Name of link
        
    Returns:
        3x3 rotation matrix representing link orientation in world frame
    """
    transform = get_link_transform(robot, joint_positions, link_name)
    return transform.rotation


def transform_point_to_link_frame(robot: RobotModel, joint_positions: np.ndarray,
                                link_name: str, world_point: np.ndarray) -> np.ndarray:
    """
    Transform a point from world frame to link frame.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        link_name: Name of target link frame
        world_point: Point in world coordinates
        
    Returns:
        Point in link frame coordinates
    """
    from ..math.transform import invert_transform, transform_point
    
    link_transform = get_link_transform(robot, joint_positions, link_name)
    inverse_transform = invert_transform(link_transform)
    return transform_point(inverse_transform, world_point)


def transform_point_to_world_frame(robot: RobotModel, joint_positions: np.ndarray,
                                 link_name: str, link_point: np.ndarray) -> np.ndarray:
    """
    Transform a point from link frame to world frame.
    
    Args:
        robot: Robot model
        joint_positions: Joint positions
        link_name: Name of source link frame
        link_point: Point in link coordinates
        
    Returns:
        Point in world frame coordinates
    """
    link_transform = get_link_transform(robot, joint_positions, link_name)
    return transform_point(link_transform, link_point)


@dataclass
class ForwardKinematics:
    """
    Object-oriented interface for forward kinematics computations.
    
    This class provides a stateful interface while using pure functions internally.
    Useful for repeated computations on the same robot model.
    """
    robot: RobotModel
    
    def compute(self, joint_positions: np.ndarray) -> KinematicState:
        """Compute forward kinematics."""
        return compute_forward_kinematics(self.robot, joint_positions)
    
    def get_link_transform(self, joint_positions: np.ndarray, link_name: str) -> Transform:
        """Get transform for specific link."""
        return get_link_transform(self.robot, joint_positions, link_name)
    
    def get_link_position(self, joint_positions: np.ndarray, link_name: str) -> np.ndarray:
        """Get position for specific link."""
        return get_link_position(self.robot, joint_positions, link_name)
    
    def get_link_orientation(self, joint_positions: np.ndarray, link_name: str) -> np.ndarray:
        """Get orientation for specific link."""
        return get_link_orientation(self.robot, joint_positions, link_name)
