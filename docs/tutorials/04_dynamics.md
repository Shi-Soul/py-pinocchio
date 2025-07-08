# Robot Dynamics in py-pinocchio

This tutorial covers robot dynamics, including the equations of motion, efficient algorithms, and control applications. You'll learn how forces and torques relate to robot motion.

## Table of Contents

1. [Equations of Motion](#equations-of-motion)
2. [Mass Matrix Properties](#mass-matrix-properties)
3. [Coriolis and Centrifugal Forces](#coriolis-and-centrifugal-forces)
4. [Gravity Compensation](#gravity-compensation)
5. [Forward and Inverse Dynamics](#forward-and-inverse-dynamics)
6. [Energy and Momentum](#energy-and-momentum)
7. [Control Applications](#control-applications)

## Equations of Motion

The fundamental equation describing robot motion is the Euler-Lagrange equation:

$$\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

Where:
- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n \times n}$ is the mass/inertia matrix
- $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) \in \mathbb{R}^{n \times n}$ captures Coriolis and centrifugal effects
- $\mathbf{g}(\mathbf{q}) \in \mathbb{R}^n$ is the gravity vector
- $\boldsymbol{\tau} \in \mathbb{R}^n$ is the applied joint torque vector
- $n$ is the number of degrees of freedom

### Derivation from Lagrangian Mechanics

The Lagrangian of a robot system is:

$$\mathcal{L} = T(\mathbf{q}, \dot{\mathbf{q}}) - V(\mathbf{q})$$

Where $T$ is kinetic energy and $V$ is potential energy. The equations of motion follow from:

$$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\mathbf{q}}}\right) - \frac{\partial \mathcal{L}}{\partial \mathbf{q}} = \boldsymbol{\tau}$$

### Practical Implementation

```python
import numpy as np
import py_pinocchio as pin

def analyze_robot_dynamics(robot, q, qd, tau):
    """
    Comprehensive analysis of robot dynamics.
    
    Args:
        robot: Robot model
        q: Joint positions
        qd: Joint velocities  
        tau: Applied torques
    """
    print("Robot Dynamics Analysis")
    print("=" * 40)
    print(f"Configuration (deg): {np.degrees(q)}")
    print(f"Velocities (deg/s): {np.degrees(qd)}")
    print(f"Applied torques (Nm): {tau}")
    
    # Compute all dynamic quantities
    M = pin.compute_mass_matrix(robot, q)
    C = pin.compute_coriolis_forces(robot, q, qd)
    g = pin.compute_gravity_vector(robot, q)
    
    print(f"\nMass Matrix M(q):")
    print(f"Shape: {M.shape}")
    print(f"Matrix:\n{M}")
    
    print(f"\nCoriolis forces C(q,qd)*qd:")
    print(f"Vector: {C}")
    
    print(f"\nGravity vector g(q):")
    print(f"Vector: {g}")
    
    # Forward dynamics: compute accelerations
    qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
    print(f"\nForward dynamics - Joint accelerations:")
    print(f"qdd (rad/s²): {qdd}")
    print(f"qdd (deg/s²): {np.degrees(qdd)}")
    
    # Inverse dynamics: compute required torques
    tau_required = pin.compute_inverse_dynamics(robot, q, qd, qdd)
    print(f"\nInverse dynamics - Required torques:")
    print(f"tau_required: {tau_required}")
    print(f"Difference from input: {np.linalg.norm(tau_required - tau):.6f}")
    
    # Energy analysis
    kinetic_energy = 0.5 * qd.T @ M @ qd
    print(f"\nEnergy Analysis:")
    print(f"Kinetic energy: {kinetic_energy:.4f} J")
    
    return {
        'mass_matrix': M,
        'coriolis_forces': C,
        'gravity_vector': g,
        'joint_accelerations': qdd,
        'required_torques': tau_required,
        'kinetic_energy': kinetic_energy
    }

# Create test robot
def create_dynamics_test_robot():
    """Create a robot for dynamics testing."""
    links = []
    joints = []
    
    # Base
    base = pin.Link("base", mass=5.0)
    links.append(base)
    
    # Two links with significant mass
    link_params = [
        {"mass": 3.0, "length": 0.5, "com_offset": 0.25},
        {"mass": 2.0, "length": 0.4, "com_offset": 0.2}
    ]
    
    for i, params in enumerate(link_params):
        # Link
        mass = params["mass"]
        length = params["length"]
        com_offset = params["com_offset"]
        
        link = pin.Link(
            name=f"link{i+1}",
            mass=mass,
            center_of_mass=np.array([com_offset, 0.0, 0.0]),
            inertia_tensor=np.array([
                [0.01, 0.0, 0.0],
                [0.0, mass * length**2 / 12, 0.0],
                [0.0, 0.0, mass * length**2 / 12]
            ])
        )
        links.append(link)
        
        # Joint
        parent = "base" if i == 0 else f"link{i}"
        offset = np.array([0, 0, 0.1]) if i == 0 else np.array([link_params[i-1]["length"], 0, 0])
        
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link=parent,
            child_link=f"link{i+1}",
            origin_transform=pin.create_transform(np.eye(3), offset)
        )
        joints.append(joint)
    
    return pin.create_robot_model("dynamics_test", links, joints, "base")

# Test dynamics
robot = create_dynamics_test_robot()
q = np.array([np.pi/6, np.pi/4])      # 30°, 45°
qd = np.array([0.5, -0.3])            # rad/s
tau = np.array([2.0, 1.0])            # Nm

dynamics_data = analyze_robot_dynamics(robot, q, qd, tau)
```

## Mass Matrix Properties

The mass matrix $\mathbf{M}(\mathbf{q})$ has several important properties:

### Mathematical Properties

1. **Symmetry**: $\mathbf{M}(\mathbf{q}) = \mathbf{M}(\mathbf{q})^T$
2. **Positive Definiteness**: $\mathbf{x}^T \mathbf{M}(\mathbf{q}) \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
3. **Configuration Dependence**: $\mathbf{M}$ varies with joint angles
4. **Bounded**: $m_{\min} \mathbf{I} \leq \mathbf{M}(\mathbf{q}) \leq m_{\max} \mathbf{I}$

### Physical Interpretation

The mass matrix represents the robot's resistance to acceleration:

$$\mathbf{M}_{ij}(\mathbf{q}) = \frac{\partial^2 T}{\partial \dot{q}_i \partial \dot{q}_j}$$

Where $T$ is the total kinetic energy.

```python
def analyze_mass_matrix_properties(robot, q_range=None, num_samples=50):
    """
    Analyze mass matrix properties across workspace.
    
    Args:
        robot: Robot model
        q_range: Range of joint angles to sample
        num_samples: Number of configurations to test
    """
    if q_range is None:
        q_range = [(-np.pi, np.pi)] * robot.num_dof
    
    print("Mass Matrix Properties Analysis")
    print("=" * 40)
    
    # Sample configurations
    eigenvalue_data = []
    condition_numbers = []
    determinants = []
    
    for _ in range(num_samples):
        # Random configuration
        q = np.array([
            np.random.uniform(low, high) 
            for low, high in q_range
        ])
        
        # Compute mass matrix
        M = pin.compute_mass_matrix(robot, q)
        
        # Analyze properties
        eigenvals = np.linalg.eigvals(M)
        eigenvalue_data.append(eigenvals)
        
        cond_num = np.linalg.cond(M)
        condition_numbers.append(cond_num)
        
        det_M = np.linalg.det(M)
        determinants.append(det_M)
        
        # Check symmetry
        symmetry_error = np.linalg.norm(M - M.T)
        if symmetry_error > 1e-12:
            print(f"Warning: Symmetry error {symmetry_error:.2e} at q={q}")
        
        # Check positive definiteness
        if np.any(eigenvals <= 0):
            print(f"Warning: Non-positive eigenvalue at q={q}")
    
    eigenvalue_data = np.array(eigenvalue_data)
    condition_numbers = np.array(condition_numbers)
    determinants = np.array(determinants)
    
    print(f"\nStatistics over {num_samples} configurations:")
    print(f"Eigenvalue range: [{eigenvalue_data.min():.4f}, {eigenvalue_data.max():.4f}]")
    print(f"Condition number range: [{condition_numbers.min():.2f}, {condition_numbers.max():.2f}]")
    print(f"Determinant range: [{determinants.min():.4f}, {determinants.max():.4f}]")
    print(f"Average condition number: {condition_numbers.mean():.2f}")
    
    # Configuration-dependent analysis
    print(f"\nConfiguration Dependence:")
    
    # Test specific configurations
    configs = {
        "home": np.zeros(robot.num_dof),
        "extended": np.array([0.0, 0.0] if robot.num_dof == 2 else [0.0] * robot.num_dof),
        "folded": np.array([np.pi/2, np.pi/2] if robot.num_dof == 2 else [np.pi/4] * robot.num_dof)
    }
    
    for config_name, q in configs.items():
        if robot.is_valid_joint_configuration(q):
            M = pin.compute_mass_matrix(robot, q)
            eigenvals = np.linalg.eigvals(M)
            cond_num = np.linalg.cond(M)
            
            print(f"  {config_name}: eigenvals={eigenvals}, cond={cond_num:.2f}")

# Analyze mass matrix
robot = create_dynamics_test_robot()
analyze_mass_matrix_properties(robot)
```

## Coriolis and Centrifugal Forces

The Coriolis matrix $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$ captures velocity-dependent forces:

$$\mathbf{C}_{ij}(\mathbf{q}, \dot{\mathbf{q}}) = \sum_{k=1}^n \Gamma_{ijk}(\mathbf{q}) \dot{q}_k$$

Where the Christoffel symbols are:

$$\Gamma_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$

### Important Property

The matrix $\dot{\mathbf{M}} - 2\mathbf{C}$ is skew-symmetric:

$$\mathbf{x}^T(\dot{\mathbf{M}} - 2\mathbf{C})\mathbf{x} = 0 \quad \forall \mathbf{x}$$

This property is crucial for stability analysis in robot control.

```python
def analyze_coriolis_properties(robot, q, qd_range=None, num_tests=20):
    """
    Analyze Coriolis force properties.
    
    Args:
        robot: Robot model
        q: Fixed joint configuration
        qd_range: Range of joint velocities
        num_tests: Number of velocity tests
    """
    print("Coriolis Forces Analysis")
    print("=" * 30)
    print(f"Configuration (deg): {np.degrees(q)}")
    
    if qd_range is None:
        qd_range = (-2.0, 2.0)  # rad/s
    
    # Test velocity dependence
    print(f"\nVelocity Dependence:")
    
    for i in range(num_tests):
        # Random velocity
        qd = np.random.uniform(qd_range[0], qd_range[1], robot.num_dof)
        
        # Compute Coriolis forces
        C_qd = pin.compute_coriolis_forces(robot, q, qd)
        
        # Test quadratic dependence
        qd_scaled = 2.0 * qd
        C_qd_scaled = pin.compute_coriolis_forces(robot, q, qd_scaled)
        
        # Should scale by factor of 4 (quadratic in velocity)
        expected_scaling = 4.0
        actual_scaling = np.linalg.norm(C_qd_scaled) / np.linalg.norm(C_qd) if np.linalg.norm(C_qd) > 1e-10 else 0
        
        if i < 5:  # Print first few tests
            print(f"  Test {i+1}: qd={qd}, C(qd)={C_qd}")
            print(f"    Scaling factor: {actual_scaling:.2f} (expected: {expected_scaling:.2f})")
    
    # Test skew-symmetry property (requires numerical differentiation)
    print(f"\nSkew-symmetry Property Test:")
    
    # Use finite differences to approximate Mdot
    dt = 1e-6
    qd_test = np.array([0.1, 0.2] if robot.num_dof == 2 else [0.1] * robot.num_dof)
    
    M1 = pin.compute_mass_matrix(robot, q)
    M2 = pin.compute_mass_matrix(robot, q + dt * qd_test)
    Mdot_approx = (M2 - M1) / dt
    
    C_matrix = compute_coriolis_matrix(robot, q, qd_test)
    
    # Check skew-symmetry of (Mdot - 2C)
    skew_matrix = Mdot_approx - 2 * C_matrix
    skew_error = np.linalg.norm(skew_matrix + skew_matrix.T)
    
    print(f"  Skew-symmetry error: {skew_error:.6f}")
    print(f"  (Should be close to zero)")

def compute_coriolis_matrix(robot, q, qd):
    """
    Compute Coriolis matrix C such that C*qd gives Coriolis forces.
    
    This is done by computing Coriolis forces for unit velocities.
    """
    n = robot.num_dof
    C = np.zeros((n, n))
    
    for i in range(n):
        qd_unit = np.zeros(n)
        qd_unit[i] = 1.0
        C[:, i] = pin.compute_coriolis_forces(robot, q, qd_unit)
    
    return C

# Test Coriolis properties
robot = create_dynamics_test_robot()
q_test = np.array([np.pi/4, np.pi/6])
analyze_coriolis_properties(robot, q_test)
```

## Gravity Compensation

Gravity forces arise from the robot's weight in a gravitational field:

$$\mathbf{g}(\mathbf{q}) = -\sum_{i=1}^n \mathbf{J}_{v,i}^T(\mathbf{q}) m_i \mathbf{g}_0$$

Where:
- $\mathbf{J}_{v,i}(\mathbf{q})$ is the linear velocity Jacobian for link $i$
- $m_i$ is the mass of link $i$  
- $\mathbf{g}_0 = [0, 0, -9.81]^T$ is the gravity vector

### Gravity Compensation Control

For static equilibrium (zero acceleration and velocity):

$$\boldsymbol{\tau} = \mathbf{g}(\mathbf{q})$$

```python
def gravity_compensation_analysis(robot):
    """
    Analyze gravity compensation across different configurations.
    
    Args:
        robot: Robot model
    """
    print("Gravity Compensation Analysis")
    print("=" * 35)
    
    # Test configurations
    test_configs = {
        "horizontal": np.array([0.0, 0.0]),
        "vertical_up": np.array([0.0, -np.pi/2]),
        "vertical_down": np.array([0.0, np.pi/2]),
        "folded": np.array([np.pi/2, np.pi/2]),
        "extended": np.array([0.0, 0.0])
    }
    
    for config_name, q in test_configs.items():
        if not robot.is_valid_joint_configuration(q):
            continue
            
        # Compute gravity compensation torques
        g_comp = pin.compute_gravity_vector(robot, q)
        
        print(f"\nConfiguration: {config_name}")
        print(f"  Joint angles (deg): {np.degrees(q)}")
        print(f"  Gravity torques (Nm): {g_comp}")
        print(f"  Total gravity effort: {np.linalg.norm(g_comp):.3f} Nm")
        
        # Verify static equilibrium
        qd_zero = np.zeros(robot.num_dof)
        qdd_zero = np.zeros(robot.num_dof)
        
        # With gravity compensation, accelerations should be zero
        qdd_with_comp = pin.compute_forward_dynamics(robot, q, qd_zero, g_comp)
        
        print(f"  Residual acceleration: {np.linalg.norm(qdd_with_comp):.6f} rad/s²")
        
        # Analyze end-effector height effect
        ee_pos = pin.get_link_position(robot, q, robot.links[-1].name)
        print(f"  End-effector height: {ee_pos[2]:.3f} m")

# Test gravity compensation
robot = create_dynamics_test_robot()
gravity_compensation_analysis(robot)
```

## Forward and Inverse Dynamics

### Forward Dynamics

Given joint torques, compute accelerations:

$$\ddot{\mathbf{q}} = \mathbf{M}(\mathbf{q})^{-1}[\boldsymbol{\tau} - \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} - \mathbf{g}(\mathbf{q})]$$

### Inverse Dynamics  

Given desired motion, compute required torques:

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

```python
def dynamics_simulation_example(robot, duration=5.0, dt=0.01):
    """
    Simulate robot dynamics with simple PD control.
    
    Args:
        robot: Robot model
        duration: Simulation time
        dt: Time step
    """
    print("Dynamics Simulation Example")
    print("=" * 30)
    
    # Simulation parameters
    steps = int(duration / dt)
    
    # Initial conditions
    q = np.array([0.1, 0.2])  # Initial position
    qd = np.zeros(robot.num_dof)  # Initial velocity
    
    # Desired trajectory (sinusoidal)
    q_desired = lambda t: np.array([0.5 * np.sin(0.5 * t), 0.3 * np.cos(0.7 * t)])
    qd_desired = lambda t: np.array([0.25 * np.cos(0.5 * t), -0.21 * np.sin(0.7 * t)])
    qdd_desired = lambda t: np.array([-0.125 * np.sin(0.5 * t), -0.147 * np.cos(0.7 * t)])
    
    # PD controller gains
    Kp = np.diag([50.0, 30.0])  # Proportional gains
    Kd = np.diag([10.0, 8.0])   # Derivative gains
    
    # Storage for results
    time_history = []
    q_history = []
    qd_history = []
    tau_history = []
    
    print(f"Simulating for {duration} seconds with dt={dt}")
    
    for step in range(steps):
        t = step * dt
        
        # Desired trajectory at current time
        qd_ref = q_desired(t)
        qdd_ref = qd_desired(t)
        qddd_ref = qdd_desired(t)
        
        # PD control law with feedforward
        q_error = qd_ref - q
        qd_error = qdd_ref - qd
        
        # Compute control torques
        tau_pd = Kp @ q_error + Kd @ qd_error
        tau_ff = pin.compute_inverse_dynamics(robot, qd_ref, qdd_ref, qddd_ref)
        tau = tau_pd + tau_ff
        
        # Forward dynamics
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        
        # Numerical integration (Euler method)
        qd += qdd * dt
        q += qd * dt
        
        # Store results
        if step % 10 == 0:  # Store every 10th step
            time_history.append(t)
            q_history.append(q.copy())
            qd_history.append(qd.copy())
            tau_history.append(tau.copy())
    
    # Convert to arrays
    time_history = np.array(time_history)
    q_history = np.array(q_history)
    qd_history = np.array(qd_history)
    tau_history = np.array(tau_history)
    
    print(f"Simulation completed. Final position: {np.degrees(q)} degrees")
    
    # Analyze results
    print(f"\nSimulation Statistics:")
    print(f"Position range (deg): [{np.degrees(q_history.min()):.1f}, {np.degrees(q_history.max()):.1f}]")
    print(f"Velocity range (deg/s): [{np.degrees(qd_history.min()):.1f}, {np.degrees(qd_history.max()):.1f}]")
    print(f"Torque range (Nm): [{tau_history.min():.2f}, {tau_history.max():.2f}]")
    
    return {
        'time': time_history,
        'positions': q_history,
        'velocities': qd_history,
        'torques': tau_history
    }

# Run simulation
robot = create_dynamics_test_robot()
sim_results = dynamics_simulation_example(robot)
```

This comprehensive dynamics tutorial covers the essential concepts and practical implementation of robot dynamics in py-pinocchio. The mathematical foundations are presented with proper LaTeX formatting, and the code examples demonstrate real-world applications.
