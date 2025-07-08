"""
MJCF (MuJoCo XML Format) parser using functional programming style

This module provides pure functions for parsing MJCF files into immutable
robot model objects. This is a simplified implementation for educational purposes.

MJCF Reference: https://mujoco.readthedocs.io/en/latest/XMLreference.html
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

from ..model import RobotModel, Link, Joint, JointType, create_robot_model
from ..math.transform import Transform, create_transform, create_identity_transform


def parse_mjcf_file(file_path: str) -> RobotModel:
    """
    Parse MJCF file and return robot model.

    Args:
        file_path: Path to MJCF file

    Returns:
        Immutable RobotModel object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If MJCF is malformed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MJCF file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        mjcf_content = f.read()

    return parse_mjcf_string(mjcf_content)


def parse_mjcf_string(mjcf_content: str) -> RobotModel:
    """
    Parse MJCF string content and return robot model.

    This is a simplified MJCF parser that handles basic MuJoCo XML format.
    It supports the most common elements for educational purposes.

    Args:
        mjcf_content: MJCF XML content as string

    Returns:
        Immutable RobotModel object

    Raises:
        ValueError: If MJCF is malformed
    """
    try:
        root = ET.fromstring(mjcf_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in MJCF: {e}")

    if root.tag != 'mujoco':
        raise ValueError("MJCF must have 'mujoco' as root element")

    model_name = root.get('model', 'mjcf_robot')

    # Parse worldbody (contains the kinematic tree)
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("MJCF must contain a 'worldbody' element")

    # Parse bodies and joints from worldbody
    links, joints = _parse_worldbody(worldbody)

    # Find root link (usually "world" or first body)
    root_link = _find_mjcf_root_link(links, joints)

    return create_robot_model(
        name=model_name,
        links=links,
        joints=joints,
        root_link=root_link
    )


def _parse_worldbody(worldbody_element: ET.Element) -> Tuple[List[Link], List[Joint]]:
    """
    Parse worldbody element to extract bodies and joints.

    In MJCF, the worldbody contains the kinematic tree structure.
    Bodies are links, and joints connect them.
    """
    links = []
    joints = []

    # Create world/base link
    world_link = Link(
        name="world",
        mass=0.0,
        center_of_mass=np.zeros(3),
        inertia_tensor=np.eye(3)
    )
    links.append(world_link)

    # Recursively parse bodies
    _parse_mjcf_bodies(worldbody_element, "world", links, joints)

    return links, joints


def _parse_mjcf_bodies(parent_element: ET.Element, parent_name: str,
                      links: List[Link], joints: List[Joint]) -> None:
    """
    Recursively parse body elements from MJCF.

    Each body becomes a link, and the connection to its parent becomes a joint.
    """
    for body_elem in parent_element.findall('body'):
        body_name = body_elem.get('name')
        if not body_name:
            body_name = f"body_{len(links)}"

        # Parse body position and orientation
        pos_str = body_elem.get('pos', '0 0 0')
        quat_str = body_elem.get('quat', '1 0 0 0')  # w x y z format

        pos = np.array([float(x) for x in pos_str.split()])
        quat = np.array([float(x) for x in quat_str.split()])

        # Convert quaternion to rotation matrix
        rotation = _quaternion_to_rotation_matrix(quat)
        origin_transform = create_transform(rotation, pos)

        # Parse inertial properties
        mass, com, inertia = _parse_mjcf_inertial(body_elem)

        # Create link
        link = Link(
            name=body_name,
            mass=mass,
            center_of_mass=com,
            inertia_tensor=inertia
        )
        links.append(link)

        # Create joint connecting to parent
        joint_elem = body_elem.find('joint')
        if joint_elem is not None:
            joint_name = joint_elem.get('name', f"joint_{body_name}")
            joint_type_str = joint_elem.get('type', 'hinge')
            axis_str = joint_elem.get('axis', '0 0 1')

            # Convert MJCF joint type to our joint type
            joint_type = _mjcf_joint_type_to_joint_type(joint_type_str)
            axis = np.array([float(x) for x in axis_str.split()])

            # Parse joint limits
            range_str = joint_elem.get('range')
            if range_str:
                range_vals = [float(x) for x in range_str.split()]
                lower_limit = range_vals[0] if len(range_vals) > 0 else -np.inf
                upper_limit = range_vals[1] if len(range_vals) > 1 else np.inf
            else:
                lower_limit = -np.inf
                upper_limit = np.inf

            joint = Joint(
                name=joint_name,
                joint_type=joint_type,
                axis=axis,
                parent_link=parent_name,
                child_link=body_name,
                origin_transform=origin_transform,
                position_limit_lower=lower_limit,
                position_limit_upper=upper_limit
            )
            joints.append(joint)
        else:
            # Fixed connection to parent
            joint = Joint(
                name=f"fixed_{body_name}",
                joint_type=JointType.FIXED,
                parent_link=parent_name,
                child_link=body_name,
                origin_transform=origin_transform
            )
            joints.append(joint)

        # Recursively parse child bodies
        _parse_mjcf_bodies(body_elem, body_name, links, joints)


def _parse_mjcf_inertial(body_element: ET.Element) -> Tuple[float, np.ndarray, np.ndarray]:
    """Parse inertial properties from MJCF body element."""
    inertial_elem = body_element.find('inertial')

    if inertial_elem is None:
        return 0.0, np.zeros(3), np.eye(3)

    # Parse mass
    mass = float(inertial_elem.get('mass', 0.0))

    # Parse center of mass position
    pos_str = inertial_elem.get('pos', '0 0 0')
    com = np.array([float(x) for x in pos_str.split()])

    # Parse inertia (MJCF uses different format than URDF)
    diaginertia_str = inertial_elem.get('diaginertia')
    if diaginertia_str:
        diag_vals = [float(x) for x in diaginertia_str.split()]
        inertia = np.diag(diag_vals)
    else:
        inertia = np.eye(3)

    return mass, com, inertia


def _quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.

    MJCF uses w-first quaternion format.
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return np.eye(3)

    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Convert to rotation matrix
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


def _mjcf_joint_type_to_joint_type(mjcf_type: str) -> JointType:
    """Convert MJCF joint type to our JointType enum."""
    type_map = {
        'hinge': JointType.REVOLUTE,
        'slide': JointType.PRISMATIC,
        'ball': JointType.REVOLUTE,  # Simplified - should be spherical
        'free': JointType.FLOATING,
    }

    return type_map.get(mjcf_type, JointType.FIXED)


def _find_mjcf_root_link(links: List[Link], joints: List[Joint]) -> str:
    """Find the root link in MJCF model (usually 'world')."""
    for link in links:
        if link.name == "world":
            return link.name

    # Fallback: find link with no parent joint
    child_links = {joint.child_link for joint in joints if joint.child_link}
    for link in links:
        if link.name not in child_links:
            return link.name

    return links[0].name if links else ""


class MJCFParser:
    """
    MJCF parser class for object-oriented interface.

    This class provides a stateful interface to MJCF parsing,
    though the underlying functions remain pure.
    """

    @staticmethod
    def parse_file(file_path: str) -> RobotModel:
        """Parse MJCF file."""
        return parse_mjcf_file(file_path)

    @staticmethod
    def parse_string(mjcf_content: str) -> RobotModel:
        """Parse MJCF string."""
        return parse_mjcf_string(mjcf_content)
