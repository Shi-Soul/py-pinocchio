"""
URDF (Unified Robot Description Format) parser using functional programming style

This module provides pure functions for parsing URDF files into immutable
robot model objects. The parser is designed for educational clarity and
handles the most common URDF elements.

URDF Reference: http://wiki.ros.org/urdf/XML
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

from ..model import RobotModel, Link, Joint, JointType, create_robot_model
from ..math.transform import Transform, create_transform, create_identity_transform
from ..math.rotation import euler_angles_to_rotation_matrix


def parse_urdf_file(file_path: str) -> RobotModel:
    """
    Parse URDF file and return robot model.
    
    Args:
        file_path: Path to URDF file
        
    Returns:
        Immutable RobotModel object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If URDF is malformed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"URDF file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()
    
    return parse_urdf_string(urdf_content)


def parse_urdf_string(urdf_content: str) -> RobotModel:
    """
    Parse URDF string content and return robot model.
    
    Args:
        urdf_content: URDF XML content as string
        
    Returns:
        Immutable RobotModel object
        
    Raises:
        ValueError: If URDF is malformed
    """
    try:
        root = ET.fromstring(urdf_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in URDF: {e}")
    
    if root.tag != 'robot':
        raise ValueError("URDF must have 'robot' as root element")
    
    robot_name = root.get('name', 'unnamed_robot')
    
    # Parse links and joints
    links = _parse_links(root)
    joints = _parse_joints(root)
    
    # Find root link (link with no parent joint)
    root_link = _find_root_link(links, joints)
    
    return create_robot_model(
        name=robot_name,
        links=links,
        joints=joints,
        root_link=root_link
    )


def _parse_links(robot_element: ET.Element) -> List[Link]:
    """
    Parse all link elements from URDF.
    
    Args:
        robot_element: Root robot XML element
        
    Returns:
        List of Link objects
    """
    links = []
    
    for link_elem in robot_element.findall('link'):
        link_name = link_elem.get('name')
        if not link_name:
            raise ValueError("Link must have a name")
        
        # Parse inertial properties
        mass, center_of_mass, inertia_tensor = _parse_inertial(link_elem)
        
        # Parse visual and collision geometry (simplified)
        visual_geometry = _parse_geometry(link_elem, 'visual')
        collision_geometry = _parse_geometry(link_elem, 'collision')
        
        link = Link(
            name=link_name,
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_tensor=inertia_tensor,
            visual_geometry=visual_geometry,
            collision_geometry=collision_geometry
        )
        
        links.append(link)
    
    return links


def _parse_joints(robot_element: ET.Element) -> List[Joint]:
    """
    Parse all joint elements from URDF.
    
    Args:
        robot_element: Root robot XML element
        
    Returns:
        List of Joint objects
    """
    joints = []
    
    for joint_elem in robot_element.findall('joint'):
        joint_name = joint_elem.get('name')
        if not joint_name:
            raise ValueError("Joint must have a name")
        
        # Parse joint type
        joint_type_str = joint_elem.get('type', 'fixed')
        joint_type = _parse_joint_type(joint_type_str)
        
        # Parse parent and child links
        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')
        
        parent_link = parent_elem.get('link') if parent_elem is not None else ""
        child_link = child_elem.get('link') if child_elem is not None else ""
        
        # Parse joint origin (transformation from parent to child)
        origin_transform = _parse_origin(joint_elem)
        
        # Parse joint axis
        axis = _parse_axis(joint_elem)
        
        # Parse joint limits
        limits = _parse_joint_limits(joint_elem)
        
        # Parse dynamics
        dynamics = _parse_joint_dynamics(joint_elem)
        
        joint = Joint(
            name=joint_name,
            joint_type=joint_type,
            axis=axis,
            parent_link=parent_link,
            child_link=child_link,
            origin_transform=origin_transform,
            position_limit_lower=limits['lower'],
            position_limit_upper=limits['upper'],
            velocity_limit=limits['velocity'],
            effort_limit=limits['effort'],
            damping=dynamics['damping'],
            friction=dynamics['friction']
        )
        
        joints.append(joint)
    
    return joints


def _parse_joint_type(type_str: str) -> JointType:
    """Parse joint type string to JointType enum."""
    type_map = {
        'revolute': JointType.REVOLUTE,
        'continuous': JointType.REVOLUTE,  # Treat continuous as revolute
        'prismatic': JointType.PRISMATIC,
        'fixed': JointType.FIXED,
        'floating': JointType.FLOATING,
        'planar': JointType.PLANAR
    }
    
    if type_str not in type_map:
        raise ValueError(f"Unsupported joint type: {type_str}")
    
    return type_map[type_str]


def _parse_inertial(link_element: ET.Element) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Parse inertial properties from link element.
    
    Returns:
        Tuple of (mass, center_of_mass, inertia_tensor)
    """
    inertial_elem = link_element.find('inertial')
    
    if inertial_elem is None:
        # No inertial properties specified
        return 0.0, np.zeros(3), np.eye(3)
    
    # Parse mass
    mass_elem = inertial_elem.find('mass')
    mass = float(mass_elem.get('value', 0.0)) if mass_elem is not None else 0.0
    
    # Parse center of mass (origin)
    origin_elem = inertial_elem.find('origin')
    center_of_mass = _parse_xyz(origin_elem) if origin_elem is not None else np.zeros(3)
    
    # Parse inertia tensor
    inertia_elem = inertial_elem.find('inertia')
    if inertia_elem is not None:
        ixx = float(inertia_elem.get('ixx', 0.0))
        iyy = float(inertia_elem.get('iyy', 0.0))
        izz = float(inertia_elem.get('izz', 0.0))
        ixy = float(inertia_elem.get('ixy', 0.0))
        ixz = float(inertia_elem.get('ixz', 0.0))
        iyz = float(inertia_elem.get('iyz', 0.0))
        
        inertia_tensor = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]
        ])
    else:
        inertia_tensor = np.eye(3)
    
    return mass, center_of_mass, inertia_tensor


def _parse_geometry(link_element: ET.Element, geom_type: str) -> Optional[Dict]:
    """
    Parse geometry information (simplified for educational purposes).
    
    Args:
        link_element: Link XML element
        geom_type: 'visual' or 'collision'
        
    Returns:
        Dictionary with geometry info or None
    """
    geom_elem = link_element.find(geom_type)
    if geom_elem is None:
        return None
    
    geometry_elem = geom_elem.find('geometry')
    if geometry_elem is None:
        return None
    
    # Simplified geometry parsing - just store the type and basic info
    geometry_info = {'type': 'unknown'}
    
    if geometry_elem.find('box') is not None:
        box_elem = geometry_elem.find('box')
        size_str = box_elem.get('size', '1 1 1')
        size = [float(x) for x in size_str.split()]
        geometry_info = {'type': 'box', 'size': size}
    
    elif geometry_elem.find('cylinder') is not None:
        cyl_elem = geometry_elem.find('cylinder')
        radius = float(cyl_elem.get('radius', 1.0))
        length = float(cyl_elem.get('length', 1.0))
        geometry_info = {'type': 'cylinder', 'radius': radius, 'length': length}
    
    elif geometry_elem.find('sphere') is not None:
        sphere_elem = geometry_elem.find('sphere')
        radius = float(sphere_elem.get('radius', 1.0))
        geometry_info = {'type': 'sphere', 'radius': radius}
    
    elif geometry_elem.find('mesh') is not None:
        mesh_elem = geometry_elem.find('mesh')
        filename = mesh_elem.get('filename', '')
        geometry_info = {'type': 'mesh', 'filename': filename}
    
    return geometry_info


def _parse_origin(element: ET.Element) -> Transform:
    """Parse origin element to Transform object."""
    origin_elem = element.find('origin')
    
    if origin_elem is None:
        return create_identity_transform()
    
    # Parse translation
    translation = _parse_xyz(origin_elem)
    
    # Parse rotation (RPY - Roll Pitch Yaw)
    rotation = _parse_rpy(origin_elem)
    
    return create_transform(rotation, translation)


def _parse_xyz(element: ET.Element) -> np.ndarray:
    """Parse xyz attribute to 3D vector."""
    xyz_str = element.get('xyz', '0 0 0')
    return np.array([float(x) for x in xyz_str.split()])


def _parse_rpy(element: ET.Element) -> np.ndarray:
    """Parse rpy attribute to rotation matrix."""
    rpy_str = element.get('rpy', '0 0 0')
    roll, pitch, yaw = [float(x) for x in rpy_str.split()]
    return euler_angles_to_rotation_matrix(roll, pitch, yaw)


def _parse_axis(joint_element: ET.Element) -> np.ndarray:
    """Parse joint axis."""
    axis_elem = joint_element.find('axis')
    if axis_elem is not None:
        xyz_str = axis_elem.get('xyz', '0 0 1')
        return np.array([float(x) for x in xyz_str.split()])
    return np.array([0, 0, 1])  # Default Z-axis


def _parse_joint_limits(joint_element: ET.Element) -> Dict[str, float]:
    """Parse joint limits."""
    limit_elem = joint_element.find('limit')
    
    if limit_elem is not None:
        return {
            'lower': float(limit_elem.get('lower', -np.inf)),
            'upper': float(limit_elem.get('upper', np.inf)),
            'velocity': float(limit_elem.get('velocity', np.inf)),
            'effort': float(limit_elem.get('effort', np.inf))
        }
    
    return {
        'lower': -np.inf,
        'upper': np.inf,
        'velocity': np.inf,
        'effort': np.inf
    }


def _parse_joint_dynamics(joint_element: ET.Element) -> Dict[str, float]:
    """Parse joint dynamics."""
    dynamics_elem = joint_element.find('dynamics')
    
    if dynamics_elem is not None:
        return {
            'damping': float(dynamics_elem.get('damping', 0.0)),
            'friction': float(dynamics_elem.get('friction', 0.0))
        }
    
    return {'damping': 0.0, 'friction': 0.0}


def _find_root_link(links: List[Link], joints: List[Joint]) -> str:
    """Find the root link (link with no parent joint)."""
    child_links = {joint.child_link for joint in joints if joint.child_link}
    
    for link in links:
        if link.name not in child_links:
            return link.name
    
    # If no root found, return first link
    return links[0].name if links else ""


class URDFParser:
    """
    URDF parser class for object-oriented interface.
    
    This class provides a stateful interface to URDF parsing,
    though the underlying functions remain pure.
    """
    
    @staticmethod
    def parse_file(file_path: str) -> RobotModel:
        """Parse URDF file."""
        return parse_urdf_file(file_path)
    
    @staticmethod
    def parse_string(urdf_content: str) -> RobotModel:
        """Parse URDF string."""
        return parse_urdf_string(urdf_content)
