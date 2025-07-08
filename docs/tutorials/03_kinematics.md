# Robot Kinematics in py-pinocchio

This tutorial provides a comprehensive guide to robot kinematics, covering forward kinematics, inverse kinematics, Jacobians, and practical applications. You'll learn both the theory and implementation details.

## Table of Contents

1. [Forward Kinematics](#forward-kinematics)
2. [Inverse Kinematics](#inverse-kinematics)
3. [Jacobian Matrices](#jacobian-matrices)
4. [Singularities and Workspace](#singularities-and-workspace)
5. [Advanced Topics](#advanced-topics)

## Forward Kinematics

Forward kinematics computes the position and orientation of robot links given joint configurations. This is the foundation of robot motion analysis.

### Mathematical Foundation

For a serial robot with $n$ joints, the forward kinematics equation is:

$$\mathbf{T}_0^n(\mathbf{q}) = \mathbf{T}_0^1(q_1) \mathbf{T}_1^2(q_2) \cdots \mathbf{T}_{n-1}^n(q_n)$$

Where:
- $\mathbf{T}_i^{i+1}(q_{i+1})$ is the transformation from frame $i$ to frame $i+1$
- $\mathbf{q} = [q_1, q_2, \ldots, q_n]^T$ is the joint configuration vector

### Implementation Example

```python
import numpy as np
import py_pinocchio as pin

# Create a 3-DOF planar robot
def create_3dof_planar_robot():
    """Create a 3-DOF planar robot for kinematics demonstration."""
    links = []
    joints = []
    
    # Link parameters
    link_lengths = [0.4, 0.3, 0.2]  # meters
    link_masses = [2.0, 1.5, 1.0]   # kg
    
    # Base link
    base = pin.Link("base", mass=3.0)
    links.append(base)
    
    # Create links and joints
    for i, (length, mass) in enumerate(zip(link_lengths, link_masses)):
        # Link
        link = pin.Link(
            name=f"link{i+1}",
            mass=mass,
            center_of_mass=np.array([length/2, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, mass*length**2/12, mass*length**2/12])
        )
        links.append(link)
        
        # Joint
        parent = "base" if i == 0 else f"link{i}"
        offset = np.array([0, 0, 0.1]) if i == 0 else np.array([link_lengths[i-1], 0, 0])
        
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),  # Z-axis rotation
            parent_link=parent,
            child_link=f"link{i+1}",
            origin_transform=pin.create_transform(np.eye(3), offset),
            position_limit_lower=-np.pi,
            position_limit_upper=np.pi
        )
        joints.append(joint)
    
    return pin.create_robot_model("3dof_planar", links, joints, "base")

# Demonstrate forward kinematics
robot = create_3dof_planar_robot()

# Define joint configurations
configurations = {
    "home": np.array([0.0, 0.0, 0.0]),
    "reach_forward": np.array([0.0, -np.pi/4, -np.pi/2]),
    "reach_up": np.array([0.0, -np.pi/2, 0.0]),
    "folded": np.array([0.0, np.pi/2, np.pi/2]),
    "side_reach": np.array([np.pi/2, -np.pi/4, -np.pi/4])
}

print("Forward Kinematics Analysis")
print("=" * 40)

for config_name, q in configurations.items():
    # Compute forward kinematics
    kinematic_state = pin.compute_forward_kinematics(robot, q)
    
    # Get end-effector pose
    ee_transform = kinematic_state.link_transforms["link3"]
    ee_position = ee_transform.translation
    ee_rotation = ee_transform.rotation
    
    print(f"\nConfiguration: {config_name}")
    print(f"Joint angles (deg): {np.degrees(q)}")
    print(f"End-effector position: [{ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f}]")
    print(f"Distance from base: {np.linalg.norm(ee_position):.3f} m")
    
    # Extract orientation (Z-Y-X Euler angles for planar robot)
    # For planar robot, only rotation about Z matters
    angle_z = np.arctan2(ee_rotation[1, 0], ee_rotation[0, 0])
    print(f"End-effector orientation (Z-rotation): {np.degrees(angle_z):.1f}°")
```

### Denavit-Hartenberg Parameters

For systematic forward kinematics, the Denavit-Hartenberg (DH) convention is widely used:

$$\mathbf{T}_i^{i+1} = \text{Rot}_z(\theta_i) \text{Trans}_z(d_i) \text{Trans}_x(a_i) \text{Rot}_x(\alpha_i)$$

```python
def dh_transform(theta, d, a, alpha):
    """
    Compute transformation matrix from DH parameters.
    
    Args:
        theta: Joint angle (rotation about z)
        d: Link offset (translation along z)
        a: Link length (translation along x)
        alpha: Link twist (rotation about x)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])
    
    return pin.Transform(T[:3, :3], T[:3, 3])

def forward_kinematics_dh(dh_params, joint_angles):
    """
    Compute forward kinematics using DH parameters.
    
    Args:
        dh_params: List of (a, alpha, d, theta_offset) tuples
        joint_angles: Current joint angles
    """
    T = pin.create_identity_transform()
    
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        T_i = dh_transform(theta, d, a, alpha)
        T = pin.math.transform.compose_transforms(T, T_i)
    
    return T

# Example: PUMA-like robot DH parameters
puma_dh = [
    (0,     np.pi/2,  0.67,  0),      # Joint 1
    (0.43,  0,        0,     0),      # Joint 2  
    (0.02,  np.pi/2,  0.15,  0),      # Joint 3
    (0,     -np.pi/2, 0.43,  0),      # Joint 4
    (0,     np.pi/2,  0,     0),      # Joint 5
    (0,     0,        0.1,   0),      # Joint 6
]

# Compute forward kinematics
q_puma = np.array([0, -np.pi/4, np.pi/2, 0, np.pi/4, 0])
T_ee = forward_kinematics_dh(puma_dh, q_puma)
print(f"PUMA end-effector position: {T_ee.translation}")
```

## Inverse Kinematics

Inverse kinematics solves for joint angles given desired end-effector pose. This is generally more challenging than forward kinematics.

### Mathematical Formulation

Given desired end-effector pose $\mathbf{T}_d$, find joint angles $\mathbf{q}$ such that:

$$\mathbf{T}_0^n(\mathbf{q}) = \mathbf{T}_d$$

This is typically a nonlinear equation that requires iterative solution methods.

### Jacobian-Based Methods

The most common approach uses the Jacobian matrix:

$$\Delta \mathbf{x} = \mathbf{J}(\mathbf{q}) \Delta \mathbf{q}$$

Where $\Delta \mathbf{x}$ is the change in end-effector pose and $\Delta \mathbf{q}$ is the change in joint angles.

#### Newton-Raphson Method

```python
def inverse_kinematics_newton_raphson(robot, target_position, target_orientation=None, 
                                    initial_guess=None, max_iterations=100, tolerance=1e-6):
    """
    Solve inverse kinematics using Newton-Raphson method.
    
    Args:
        robot: Robot model
        target_position: Desired end-effector position (3D)
        target_orientation: Desired orientation (3x3 matrix, optional)
        initial_guess: Initial joint configuration
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
    """
    if initial_guess is None:
        q = np.zeros(robot.num_dof)
    else:
        q = np.array(initial_guess)
    
    target_pos = np.array(target_position)
    
    for iteration in range(max_iterations):
        # Current end-effector pose
        current_pos = pin.get_link_position(robot, q, robot.links[-1].name)
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Check convergence
        if np.linalg.norm(pos_error) < tolerance:
            return q, True, iteration
        
        # Compute Jacobian (position part only for position-only IK)
        J_full = pin.compute_geometric_jacobian(robot, q, robot.links[-1].name)
        J_pos = J_full[3:6, :]  # Linear velocity part
        
        # Compute pseudoinverse
        try:
            J_pinv = np.linalg.pinv(J_pos)
        except np.linalg.LinAlgError:
            return q, False, iteration
        
        # Update joint angles
        dq = J_pinv @ pos_error
        
        # Apply step size for stability
        step_size = 0.1
        q += step_size * dq
        
        # Enforce joint limits
        for i, joint_name in enumerate(robot.actuated_joint_names):
            joint = robot.get_joint(joint_name)
            if joint:
                q[i] = np.clip(q[i], joint.position_limit_lower, joint.position_limit_upper)
    
    return q, False, max_iterations

# Example usage
robot = create_3dof_planar_robot()
target_pos = np.array([0.6, 0.4, 0.1])

q_solution, converged, iterations = inverse_kinematics_newton_raphson(
    robot, target_pos, initial_guess=np.array([0.5, -0.5, 0.5])
)

if converged:
    print(f"IK converged in {iterations} iterations")
    print(f"Solution: {np.degrees(q_solution)} degrees")
    
    # Verify solution
    actual_pos = pin.get_link_position(robot, q_solution, "link3")
    error = np.linalg.norm(actual_pos - target_pos)
    print(f"Position error: {error:.6f} m")
else:
    print(f"IK failed to converge after {iterations} iterations")
```

#### Damped Least Squares Method

For better numerical stability near singularities:

$$\Delta \mathbf{q} = (\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I})^{-1} \mathbf{J}^T \Delta \mathbf{x}$$

```python
def inverse_kinematics_damped_ls(robot, target_position, damping_factor=0.01, 
                               initial_guess=None, max_iterations=100, tolerance=1e-6):
    """
    Solve inverse kinematics using damped least squares method.
    
    Args:
        robot: Robot model
        target_position: Desired end-effector position
        damping_factor: Damping parameter (lambda)
        initial_guess: Initial joint configuration
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    """
    if initial_guess is None:
        q = np.zeros(robot.num_dof)
    else:
        q = np.array(initial_guess)
    
    target_pos = np.array(target_position)
    
    for iteration in range(max_iterations):
        # Current end-effector position
        current_pos = pin.get_link_position(robot, q, robot.links[-1].name)
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Check convergence
        if np.linalg.norm(pos_error) < tolerance:
            return q, True, iteration
        
        # Compute Jacobian
        J_full = pin.compute_geometric_jacobian(robot, q, robot.links[-1].name)
        J_pos = J_full[3:6, :]
        
        # Damped least squares solution
        JTJ = J_pos.T @ J_pos
        damped_inverse = np.linalg.inv(JTJ + damping_factor * np.eye(robot.num_dof))
        dq = damped_inverse @ J_pos.T @ pos_error
        
        # Update with adaptive step size
        step_size = min(0.1, 1.0 / (1.0 + iteration * 0.01))
        q += step_size * dq
        
        # Enforce joint limits
        for i, joint_name in enumerate(robot.actuated_joint_names):
            joint = robot.get_joint(joint_name)
            if joint:
                q[i] = np.clip(q[i], joint.position_limit_lower, joint.position_limit_upper)
    
    return q, False, max_iterations

# Compare methods
target_pos = np.array([0.5, 0.3, 0.1])

print("Comparing IK Methods")
print("=" * 30)

# Newton-Raphson
q_nr, conv_nr, iter_nr = inverse_kinematics_newton_raphson(robot, target_pos)
pos_nr = pin.get_link_position(robot, q_nr, "link3")
error_nr = np.linalg.norm(pos_nr - target_pos)

print(f"Newton-Raphson: converged={conv_nr}, iterations={iter_nr}, error={error_nr:.6f}")

# Damped Least Squares
q_dls, conv_dls, iter_dls = inverse_kinematics_damped_ls(robot, target_pos)
pos_dls = pin.get_link_position(robot, q_dls, "link3")
error_dls = np.linalg.norm(pos_dls - target_pos)

print(f"Damped LS: converged={conv_dls}, iterations={iter_dls}, error={error_dls:.6f}")
```

### Analytical Solutions

For specific robot geometries, analytical solutions exist and are much faster:

```python
def analytical_ik_2dof_planar(L1, L2, target_x, target_y):
    """
    Analytical inverse kinematics for 2-DOF planar robot.
    
    Args:
        L1, L2: Link lengths
        target_x, target_y: Target position
    
    Returns:
        List of valid joint angle solutions
    """
    # Distance to target
    r = np.sqrt(target_x**2 + target_y**2)
    
    # Check reachability
    if r > L1 + L2 or r < abs(L1 - L2):
        return []  # Target unreachable
    
    # Angle to target
    phi = np.arctan2(target_y, target_x)
    
    # Law of cosines for elbow angle
    cos_q2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1, 1)  # Numerical safety
    
    # Two solutions (elbow up/down)
    q2_1 = np.arccos(cos_q2)
    q2_2 = -np.arccos(cos_q2)
    
    solutions = []
    
    for q2 in [q2_1, q2_2]:
        # Shoulder angle
        alpha = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = phi - alpha
        
        solutions.append([q1, q2])
    
    return solutions

# Example usage
L1, L2 = 0.4, 0.3
target_x, target_y = 0.5, 0.3

solutions = analytical_ik_2dof_planar(L1, L2, target_x, target_y)

print(f"Analytical IK solutions for target ({target_x}, {target_y}):")
for i, sol in enumerate(solutions):
    print(f"Solution {i+1}: q1={np.degrees(sol[0]):.1f}°, q2={np.degrees(sol[1]):.1f}°")
    
    # Verify solution
    x = L1 * np.cos(sol[0]) + L2 * np.cos(sol[0] + sol[1])
    y = L1 * np.sin(sol[0]) + L2 * np.sin(sol[0] + sol[1])
    error = np.sqrt((x - target_x)**2 + (y - target_y)**2)
    print(f"  Verification error: {error:.6f}")
```

## Jacobian Matrices

The Jacobian matrix is fundamental to robot kinematics and control. It relates joint velocities to end-effector velocities.

### Mathematical Definition

The geometric Jacobian is defined as:

$$\mathbf{v} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

Where $\mathbf{v} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}$ is the 6D spatial velocity.

### Jacobian Computation

```python
def analyze_jacobian_properties(robot, q):
    """
    Analyze properties of the Jacobian matrix.
    
    Args:
        robot: Robot model
        q: Joint configuration
    """
    # Compute Jacobian
    J = pin.compute_geometric_jacobian(robot, q, robot.links[-1].name)
    
    print(f"Jacobian Analysis for configuration: {np.degrees(q)}")
    print("=" * 50)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian matrix:\n{J}")
    
    # Decompose into angular and linear parts
    J_angular = J[:3, :]  # Angular velocity part
    J_linear = J[3:, :]   # Linear velocity part
    
    print(f"\nAngular Jacobian:\n{J_angular}")
    print(f"\nLinear Jacobian:\n{J_linear}")
    
    # Compute important properties
    
    # 1. Rank
    rank = np.linalg.matrix_rank(J)
    print(f"\nJacobian rank: {rank}")
    
    # 2. Condition number
    cond_num = np.linalg.cond(J)
    print(f"Condition number: {cond_num:.2f}")
    
    # 3. Manipulability (Yoshikawa measure)
    if J.shape[0] == J.shape[1]:  # Square matrix
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
    else:
        # Use linear part for manipulability
        manipulability = np.sqrt(np.linalg.det(J_linear @ J_linear.T))
    print(f"Manipulability: {manipulability:.6f}")
    
    # 4. Singular values
    U, s, Vt = np.linalg.svd(J)
    print(f"Singular values: {s}")
    
    # 5. Null space (for redundant robots)
    if J.shape[1] > J.shape[0]:  # More DOF than task space
        null_space_dim = J.shape[1] - rank
        print(f"Null space dimension: {null_space_dim}")
        
        if null_space_dim > 0:
            # Compute null space basis
            _, _, Vt = np.linalg.svd(J)
            null_space = Vt[rank:, :].T
            print(f"Null space basis:\n{null_space}")
    
    # 6. Check for singularities
    is_singular = cond_num > 1000  # Arbitrary threshold
    print(f"\nNear singularity: {is_singular}")
    
    return {
        'jacobian': J,
        'rank': rank,
        'condition_number': cond_num,
        'manipulability': manipulability,
        'singular_values': s,
        'is_singular': is_singular
    }

# Analyze Jacobian at different configurations
robot = create_3dof_planar_robot()

test_configs = [
    ("Home", np.array([0.0, 0.0, 0.0])),
    ("Stretched", np.array([0.0, 0.0, 0.0])),
    ("Folded", np.array([0.0, np.pi/2, np.pi/2])),
    ("Singular", np.array([0.0, 0.0, np.pi])),  # Fully extended
]

for config_name, q in test_configs:
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    properties = analyze_jacobian_properties(robot, q)
```

This tutorial provides a solid foundation for understanding robot kinematics. The next sections would cover singularities, workspace analysis, and advanced topics like differential kinematics and motion planning.
