"""
Robot model representation using functional programming style

This module defines immutable data structures for representing robot models,
including joints, links, and complete kinematic chains. The design emphasizes
clarity and educational value over performance.

All data structures are immutable and operations are pure functions.
"""

import numpy as np
from typing import List, Dict, Optional, NamedTuple, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .math.transform import Transform, create_identity_transform
from .math.spatial import SpatialInertia, create_spatial_inertia_matrix


class JointType(Enum):
    """Enumeration of supported joint types."""
    REVOLUTE = "revolute"      # 1-DOF rotation about axis
    PRISMATIC = "prismatic"    # 1-DOF translation along axis
    FIXED = "fixed"            # 0-DOF fixed connection
    FLOATING = "floating"      # 6-DOF free motion (for base)
    PLANAR = "planar"         # 3-DOF planar motion (2 translation + 1 rotation)


@dataclass(frozen=True)
class Joint:
    """
    Immutable joint representation.
    
    A joint connects two links and defines their relative motion constraints.
    Each joint has a type, axis of motion, and limits.
    """
    name: str
    joint_type: JointType
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    parent_link: str = ""
    child_link: str = ""
    origin_transform: Transform = field(default_factory=create_identity_transform)
    
    # Joint limits
    position_limit_lower: float = -np.inf
    position_limit_upper: float = np.inf
    velocity_limit: float = np.inf
    effort_limit: float = np.inf
    
    # Physical properties
    damping: float = 0.0
    friction: float = 0.0
    
    def __post_init__(self):
        """Validate joint parameters."""
        if not isinstance(self.joint_type, JointType):
            raise ValueError("joint_type must be JointType enum")
        
        axis = np.asarray(self.axis)
        if axis.shape != (3,):
            raise ValueError("Joint axis must be 3D vector")
        
        # Normalize axis for revolute and prismatic joints
        if self.joint_type in [JointType.REVOLUTE, JointType.PRISMATIC]:
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-12:
                raise ValueError("Joint axis cannot be zero vector")
            object.__setattr__(self, 'axis', axis / axis_norm)
    
    @property
    def degrees_of_freedom(self) -> int:
        """Get number of degrees of freedom for this joint."""
        dof_map = {
            JointType.FIXED: 0,
            JointType.REVOLUTE: 1,
            JointType.PRISMATIC: 1,
            JointType.PLANAR: 3,
            JointType.FLOATING: 6
        }
        return dof_map[self.joint_type]
    
    @property
    def is_actuated(self) -> bool:
        """Check if joint is actuated (has DOF > 0)."""
        return self.degrees_of_freedom > 0


@dataclass(frozen=True)
class Link:
    """
    Immutable link representation.
    
    A link is a rigid body with mass, inertia, and geometric properties.
    Links are connected by joints to form kinematic chains.
    """
    name: str
    
    # Inertial properties
    mass: float = 0.0
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    # Geometric properties (for visualization and collision)
    visual_geometry: Optional[Dict] = None
    collision_geometry: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate link parameters."""
        if self.mass < 0:
            raise ValueError("Link mass cannot be negative")
        
        com = np.asarray(self.center_of_mass)
        if com.shape != (3,):
            raise ValueError("Center of mass must be 3D vector")
        object.__setattr__(self, 'center_of_mass', com.copy())
        
        inertia = np.asarray(self.inertia_tensor)
        if inertia.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3x3 matrix")
        object.__setattr__(self, 'inertia_tensor', inertia.copy())
    
    @property
    def spatial_inertia(self) -> SpatialInertia:
        """Get spatial inertia matrix for this link."""
        return create_spatial_inertia_matrix(
            mass=self.mass,
            center_of_mass=self.center_of_mass,
            inertia_tensor=self.inertia_tensor
        )
    
    @property
    def has_inertia(self) -> bool:
        """Check if link has non-zero inertia."""
        return self.mass > 1e-12


@dataclass(frozen=True)
class RobotModel:
    """
    Immutable robot model representation.
    
    A complete robot model consists of links connected by joints in a tree structure.
    The model provides methods for kinematic and dynamic computations.
    """
    name: str
    links: Tuple[Link, ...] = field(default_factory=tuple)
    joints: Tuple[Joint, ...] = field(default_factory=tuple)
    root_link: str = ""
    
    # Derived properties (computed once during construction)
    _link_map: Dict[str, Link] = field(default_factory=dict, init=False)
    _joint_map: Dict[str, Joint] = field(default_factory=dict, init=False)
    _parent_map: Dict[str, str] = field(default_factory=dict, init=False)  # child -> parent
    _children_map: Dict[str, List[str]] = field(default_factory=dict, init=False)  # parent -> children
    _joint_order: Tuple[str, ...] = field(default_factory=tuple, init=False)
    
    def __post_init__(self):
        """Build internal maps and validate model structure."""
        # Build link and joint maps
        link_map = {link.name: link for link in self.links}
        joint_map = {joint.name: joint for joint in self.joints}
        
        object.__setattr__(self, '_link_map', link_map)
        object.__setattr__(self, '_joint_map', joint_map)
        
        # Validate root link exists
        if self.root_link and self.root_link not in link_map:
            raise ValueError(f"Root link '{self.root_link}' not found in links")
        
        # Build parent-child relationships
        parent_map = {}
        children_map = {link.name: [] for link in self.links}
        
        for joint in self.joints:
            if joint.parent_link and joint.child_link:
                if joint.parent_link not in link_map:
                    raise ValueError(f"Parent link '{joint.parent_link}' not found")
                if joint.child_link not in link_map:
                    raise ValueError(f"Child link '{joint.child_link}' not found")
                
                parent_map[joint.child_link] = joint.parent_link
                children_map[joint.parent_link].append(joint.child_link)
        
        object.__setattr__(self, '_parent_map', parent_map)
        object.__setattr__(self, '_children_map', children_map)
        
        # Determine joint order (topological sort)
        joint_order = self._compute_joint_order()
        object.__setattr__(self, '_joint_order', joint_order)
    
    def _compute_joint_order(self) -> Tuple[str, ...]:
        """Compute topological ordering of joints from root to leaves."""
        if not self.root_link:
            return tuple()
        
        joint_order = []
        visited = set()
        
        def visit_link(link_name: str):
            if link_name in visited:
                return
            visited.add(link_name)
            
            # Visit all child links
            for child_link in self._children_map.get(link_name, []):
                # Find joint connecting parent to child
                joint = self.get_joint_between_links(link_name, child_link)
                if joint and joint.name not in [j for j in joint_order]:
                    joint_order.append(joint.name)
                visit_link(child_link)
        
        visit_link(self.root_link)
        return tuple(joint_order)
    
    @property
    def num_links(self) -> int:
        """Get number of links in the model."""
        return len(self.links)
    
    @property
    def num_joints(self) -> int:
        """Get number of joints in the model."""
        return len(self.joints)
    
    @property
    def num_dof(self) -> int:
        """Get total degrees of freedom (number of actuated joints)."""
        return sum(joint.degrees_of_freedom for joint in self.joints if joint.is_actuated)
    
    @property
    def actuated_joint_names(self) -> List[str]:
        """Get names of actuated joints in kinematic order."""
        return [name for name in self._joint_order 
                if name in self._joint_map and self._joint_map[name].is_actuated]
    
    def get_link(self, name: str) -> Optional[Link]:
        """Get link by name."""
        return self._link_map.get(name)
    
    def get_joint(self, name: str) -> Optional[Joint]:
        """Get joint by name."""
        return self._joint_map.get(name)
    
    def get_joint_between_links(self, parent_link: str, child_link: str) -> Optional[Joint]:
        """Find joint connecting two links."""
        for joint in self.joints:
            if joint.parent_link == parent_link and joint.child_link == child_link:
                return joint
        return None
    
    def get_parent_link(self, link_name: str) -> Optional[str]:
        """Get parent link name."""
        return self._parent_map.get(link_name)
    
    def get_child_links(self, link_name: str) -> List[str]:
        """Get child link names."""
        return self._children_map.get(link_name, [])
    
    def is_valid_joint_configuration(self, joint_positions: np.ndarray) -> bool:
        """Check if joint configuration is within limits."""
        if len(joint_positions) != self.num_dof:
            return False
        
        actuated_joints = [self._joint_map[name] for name in self.actuated_joint_names]
        
        for i, joint in enumerate(actuated_joints):
            pos = joint_positions[i]
            if not (joint.position_limit_lower <= pos <= joint.position_limit_upper):
                return False
        
        return True


def create_robot_model(name: str, links: List[Link], joints: List[Joint], 
                      root_link: str = "") -> RobotModel:
    """
    Create robot model from links and joints.
    
    Args:
        name: Robot model name
        links: List of links
        joints: List of joints
        root_link: Name of root link (base)
        
    Returns:
        Immutable RobotModel object
    """
    return RobotModel(
        name=name,
        links=tuple(links),
        joints=tuple(joints),
        root_link=root_link
    )
