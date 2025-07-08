# File I/O API Reference

This document provides comprehensive API documentation for py-pinocchio's file parsing capabilities, including URDF and MJCF format support.

## Table of Contents

1. [URDF Parser](#urdf-parser)
2. [MJCF Parser](#mjcf-parser)
3. [Convenience Functions](#convenience-functions)
4. [Error Handling](#error-handling)
5. [Supported Features](#supported-features)

## URDF Parser

### `URDFParser`

Parser class for URDF (Unified Robot Description Format) files.

```python
class URDFParser:
    """Parser for URDF robot description files."""
    
    def __init__(self, validate_xml: bool = True):
        """Initialize URDF parser."""
```

#### Constructor Parameters

- **`validate_xml`** (`bool`, optional): Enable XML validation. Default: `True`

#### Methods

##### `parse_file(filepath: str) -> RobotModel`

Parse URDF file and create robot model.

**Parameters:**
- `filepath`: Path to URDF file

**Returns:**
- `RobotModel`: Parsed robot model

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `XMLSyntaxError`: If XML is malformed
- `URDFParseError`: If URDF content is invalid

**Example:**
```python
import py_pinocchio as pin

parser = pin.URDFParser()
robot = parser.parse_file("robot.urdf")
print(f"Loaded robot: {robot.name} with {robot.num_dof} DOF")
```

##### `parse_string(urdf_content: str) -> RobotModel`

Parse URDF from string content.

**Parameters:**
- `urdf_content`: URDF XML content as string

**Returns:**
- `RobotModel`: Parsed robot model

##### `validate_urdf(filepath: str) -> bool`

Validate URDF file without parsing.

**Parameters:**
- `filepath`: Path to URDF file

**Returns:**
- `bool`: True if valid, False otherwise

### Convenience Functions

#### `parse_urdf_file(filepath: str, validate: bool = True) -> RobotModel`

Quick function to parse URDF file.

**Parameters:**
- `filepath`: Path to URDF file
- `validate`: Enable validation (default: True)

**Returns:**
- `RobotModel`: Parsed robot model

**Example:**
```python
# Simple URDF loading
robot = pin.parse_urdf_file("my_robot.urdf")
```

#### `parse_urdf_string(urdf_content: str, validate: bool = True) -> RobotModel`

Quick function to parse URDF from string.

**Parameters:**
- `urdf_content`: URDF XML content
- `validate`: Enable validation (default: True)

**Returns:**
- `RobotModel`: Parsed robot model

## MJCF Parser

### `MJCFParser`

Parser class for MJCF (MuJoCo XML Format) files.

```python
class MJCFParser:
    """Parser for MJCF robot description files."""
    
    def __init__(self, validate_xml: bool = True):
        """Initialize MJCF parser."""
```

#### Constructor Parameters

- **`validate_xml`** (`bool`, optional): Enable XML validation. Default: `True`

#### Methods

##### `parse_file(filepath: str) -> RobotModel`

Parse MJCF file and create robot model.

**Parameters:**
- `filepath`: Path to MJCF file

**Returns:**
- `RobotModel`: Parsed robot model

**Example:**
```python
parser = pin.MJCFParser()
robot = parser.parse_file("robot.xml")
```

##### `parse_string(mjcf_content: str) -> RobotModel`

Parse MJCF from string content.

**Parameters:**
- `mjcf_content`: MJCF XML content as string

**Returns:**
- `RobotModel`: Parsed robot model

### Convenience Functions

#### `parse_mjcf_file(filepath: str, validate: bool = True) -> RobotModel`

Quick function to parse MJCF file.

**Parameters:**
- `filepath`: Path to MJCF file
- `validate`: Enable validation (default: True)

**Returns:**
- `RobotModel`: Parsed robot model

#### `parse_mjcf_string(mjcf_content: str, validate: bool = True) -> RobotModel`

Quick function to parse MJCF from string.

**Parameters:**
- `mjcf_content`: MJCF XML content
- `validate`: Enable validation (default: True)

**Returns:**
- `RobotModel`: Parsed robot model

## Error Handling

### Exception Types

The file I/O module defines several specific exception types:

```python
class ParseError(Exception):
    """Base class for parsing errors."""
    pass

class URDFParseError(ParseError):
    """URDF-specific parsing error."""
    pass

class MJCFParseError(ParseError):
    """MJCF-specific parsing error."""
    pass

class XMLValidationError(ParseError):
    """XML validation error."""
    pass
```

### Error Handling Example

```python
import py_pinocchio as pin

try:
    robot = pin.parse_urdf_file("robot.urdf")
except FileNotFoundError:
    print("URDF file not found")
except pin.URDFParseError as e:
    print(f"URDF parsing failed: {e}")
except pin.XMLValidationError as e:
    print(f"XML validation failed: {e}")
```

## Supported Features

### URDF Support

**Fully Supported:**
- `<robot>` root element
- `<link>` elements with inertial properties
- `<joint>` elements (revolute, prismatic, fixed, continuous)
- `<origin>` transformations
- `<limit>` joint limits
- `<dynamics>` damping and friction
- `<material>` basic material properties

**Partially Supported:**
- `<visual>` geometry (basic shapes only)
- `<collision>` geometry (basic shapes only)
- `<xacro>` macros (basic substitution)

**Not Supported:**
- Complex mesh geometries
- Advanced material properties
- Sensors and actuators
- Gazebo-specific extensions

### MJCF Support

**Fully Supported:**
- `<mujoco>` root element
- `<worldbody>` and `<body>` hierarchy
- `<joint>` elements (hinge, slide, free, ball)
- `<geom>` basic geometries
- `<inertial>` properties
- `<default>` classes

**Partially Supported:**
- `<actuator>` elements (position/velocity control)
- `<sensor>` elements (basic types)
- `<contact>` properties

**Not Supported:**
- Advanced contact modeling
- Tendons and constraints
- Visual effects and rendering
- MuJoCo-specific physics settings

## File Format Examples

### Minimal URDF

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
</robot>
```

### Minimal MJCF

```xml
<mujoco model="simple_robot">
  <worldbody>
    <body name="base">
      <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
      <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      
      <body name="link1" pos="0.2 0 0">
        <inertial mass="0.5" diaginertia="0.05 0.05 0.05"/>
        <geom type="cylinder" size="0.05 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
```

## Requirements

The file I/O module requires:
- `lxml >= 4.6.0` (XML parsing)
- `numpy >= 1.20.0` (numerical operations)

## Notes

- Both parsers support relative file paths and package:// URIs
- XML validation can be disabled for performance in trusted environments
- Parsing preserves original file structure information for debugging
- Both formats are converted to the same internal `RobotModel` representation
