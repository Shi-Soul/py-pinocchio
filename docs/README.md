# py-pinocchio Documentation

Welcome to the comprehensive documentation for py-pinocchio! This library provides educational implementations of rigid body dynamics algorithms for robotics, designed to help you understand the fundamental concepts through clear, well-documented code.

## 🚀 Quick Start

```python
import py_pinocchio as pin
import numpy as np

# Create a simple 2-DOF robot arm
base = pin.Link("base", mass=1.0)
link1 = pin.Link("link1", mass=2.0, center_of_mass=np.array([0.25, 0, 0]))
link2 = pin.Link("link2", mass=1.5, center_of_mass=np.array([0.15, 0, 0]))

joint1 = pin.Joint("joint1", pin.JointType.REVOLUTE, axis=np.array([0, 0, 1]),
                   parent_link="base", child_link="link1")
joint2 = pin.Joint("joint2", pin.JointType.REVOLUTE, axis=np.array([0, 0, 1]),
                   parent_link="link1", child_link="link2",
                   origin_transform=pin.create_transform(np.eye(3), np.array([0.5, 0, 0])))

robot = pin.create_robot_model("simple_arm", [base, link1, link2], [joint1, joint2], "base")

# Compute forward kinematics
q = np.array([np.pi/4, np.pi/6])  # 45° and 30°
end_effector_pos = pin.get_link_position(robot, q, "link2")
print(f"End-effector position: {end_effector_pos}")

# Compute dynamics
M = pin.compute_mass_matrix(robot, q)
g = pin.compute_gravity_vector(robot, q)
print(f"Mass matrix shape: {M.shape}")
print(f"Gravity forces: {g}")
```

## 📚 Documentation Structure

### 🎓 Tutorials (Start Here!)
Step-by-step learning guides for mastering py-pinocchio:

- **[Getting Started](tutorials/01_getting_started.md)** - Your first robot and basic operations
- **[Robot Modeling](tutorials/02_robot_modeling.md)** - Advanced robot creation techniques
- **[Kinematics](tutorials/03_kinematics.md)** - Forward and inverse kinematics
- **[Dynamics](tutorials/04_dynamics.md)** - Robot dynamics and control
- **[Visualization](tutorials/05_visualization.md)** - Plotting and animating robots
- **[File Formats](tutorials/06_file_formats.md)** - Working with URDF and MJCF files

### 🤖 Examples (Learn by Doing!)
Complete working examples for different robot types:

- **[Basic Usage](../examples/basic_usage.py)** - Simple robot creation and analysis
- **[Legged Robots](../examples/legged_robot_example.py)** - Bipedal robot modeling
- **[Multi-DOF Arms](../examples/multi_dof_arm_example.py)** - 6-DOF manipulator
- **[Advanced Quadruped](../examples/advanced_quadruped_example.py)** - Sophisticated 12-DOF quadruped with gait generation
- **[Humanoid Robot](../examples/humanoid_robot_example.py)** - Full-body humanoid with 30+ DOF
- **[Industrial Manipulator](../examples/industrial_manipulator_example.py)** - 6-DOF industrial robot with trajectory planning
- **[Visualization](../examples/visualization_example.py)** - Robot plotting and animation

### 📖 API Reference (Look It Up!)
Detailed function and class documentation:

- **[Core API](api/core_api.md)** - Main functions and classes
- **[Math Utilities](api/math_api.md)** - Mathematical operations and transformations
- **[Visualization API](api/visualization_api.md)** - Plotting and animation functions
- **[File I/O API](api/file_io_api.md)** - URDF and MJCF parsing

### 🧮 Theory (Understand the Math!)
Mathematical foundations and algorithms:

- **[Mathematical Foundations](theory/mathematical_foundations.md)** - Spatial algebra and rigid body dynamics
- **[Kinematic Algorithms](theory/kinematic_algorithms.md)** - Forward and inverse kinematics theory
- **[Dynamic Algorithms](theory/dynamic_algorithms.md)** - Efficient dynamics computation
- **[Numerical Methods](theory/numerical_methods.md)** - Stability and accuracy considerations

### 📋 Guides (Best Practices!)
Practical guides for effective usage:

- **[Performance Guide](guides/performance_guide.md)** - Optimization and benchmarking
- **[Testing Guide](guides/testing_guide.md)** - Writing and running tests
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to the project

## Key Features

- **Educational Focus**: Code optimized for understanding over performance
- **Functional Programming**: Pure functions with immutable data structures
- **Complete Dynamics**: Forward and inverse dynamics with spatial algebra
- **Multi-Format Support**: URDF and MJCF file parsing
- **Visualization**: 2D/3D robot plotting and animation
- **Comprehensive Examples**: From simple arms to legged robots
- **Type Safety**: Full type annotations for better development experience

## Design Philosophy

py-pinocchio is designed as an educational tool for learning rigid body dynamics in robotics. The implementation prioritizes:

1. **Clarity**: Every algorithm is clearly documented with mathematical background
2. **Correctness**: Proper implementation of robotics fundamentals
3. **Extensibility**: Easy to modify and extend for research and learning
4. **Functional Style**: Pure functions make debugging and testing easier

## Installation

```bash
pip install py-pinocchio
```

Or for development:

```bash
git clone https://github.com/your-repo/py-pinocchio
cd py-pinocchio
pip install -e .
```

## Dependencies

- `numpy>=1.20.0` - Numerical computations
- `lxml>=4.6.0` - XML parsing for URDF/MJCF
- `matplotlib>=3.5.0` - Visualization (optional)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Citation

If you use py-pinocchio in your research or education, please cite:

```bibtex
@software{py_pinocchio,
  title={py-pinocchio: Educational Rigid Body Dynamics for Robotics},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/py-pinocchio}
}
```

## Support

- **Documentation**: This documentation site
- **Examples**: Comprehensive examples in the `examples/` directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions and share ideas on GitHub Discussions

## Related Projects

- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) - The original high-performance C++ library
- [PyBullet](https://pybullet.org/) - Physics simulation for robotics
- [RigidBodyDynamics.jl](https://github.com/JuliaRobotics/RigidBodyDynamics.jl) - Julia implementation
- [Drake](https://drake.mit.edu/) - Manipulation planning and control

## Acknowledgments

This educational implementation is inspired by the excellent work of the Pinocchio team and the broader robotics community. Special thanks to all contributors to open-source robotics software.
