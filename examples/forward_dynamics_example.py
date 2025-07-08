#!/usr/bin/env python3
"""
Forward Dynamics example for py-pinocchio

This example demonstrates forward dynamics computation - given joint torques,
compute the resulting joint accelerations. This is fundamental for robot
simulation and control.

Forward dynamics solves: q̈ = M(q)^(-1) * (τ - C(q,q̇)q̇ - g(q))
where:
- q̈: joint accelerations (what we want to find)
- M(q): mass matrix (configuration-dependent inertia)
- τ: applied joint torques (input)
- C(q,q̇)q̇: Coriolis and centrifugal forces
- g(q): gravitational forces
"""

import numpy as np
import py_pinocchio as pin


def create_pendulum_robot():
    """
    Create a simple pendulum robot for forward dynamics demonstration.
    
    This is a 1-DOF system that clearly shows gravitational and inertial effects.
    """
    print("Creating simple pendulum robot...")
    
    # Create links
    base_link = pin.Link(
        name="base",
        mass=0.0,  # Base has no mass
        center_of_mass=np.zeros(3),
        inertia_tensor=np.eye(3)
    )
    
    pendulum_link = pin.Link(
        name="pendulum",
        mass=1.0,  # 1 kg mass
        center_of_mass=np.array([0.0, 0.0, -0.5]),  # COM at middle of 1m rod
        inertia_tensor=np.diag([0.083, 0.083, 0.001])  # Rod inertia
    )
    
    # Create revolute joint (rotation about Y-axis)
    pendulum_joint = pin.Joint(
        name="pendulum_joint",
        joint_type=pin.JointType.REVOLUTE,
        axis=np.array([0, 1, 0]),  # Rotate about Y-axis
        parent_link="base",
        child_link="pendulum",
        origin_transform=pin.create_identity_transform(),
        position_limit_lower=-4*np.pi,  # Allow multiple rotations
        position_limit_upper=4*np.pi
    )
    
    robot = pin.create_robot_model(
        name="pendulum",
        links=[base_link, pendulum_link],
        joints=[pendulum_joint],
        root_link="base"
    )
    
    print(f"Created pendulum with {robot.num_dof} DOF")
    return robot


def demonstrate_forward_dynamics_basics(robot):
    """Demonstrate basic forward dynamics concepts."""
    print("\n=== Forward Dynamics Basics ===")
    
    # Test different configurations and torques
    test_cases = [
        {"q": [0.0], "qd": [0.0], "tau": [0.0], "description": "At rest, no torque"},
        {"q": [np.pi/4], "qd": [0.0], "tau": [0.0], "description": "45° position, no torque (gravity effect)"},
        {"q": [0.0], "qd": [1.0], "tau": [0.0], "description": "Moving at bottom, no torque (Coriolis effect)"},
        {"q": [0.0], "qd": [0.0], "tau": [1.0], "description": "At rest, 1 N⋅m torque applied"},
        {"q": [np.pi/2], "qd": [0.0], "tau": [0.0], "description": "Horizontal position, no torque"},
    ]
    
    for i, case in enumerate(test_cases):
        q = np.array(case["q"])
        qd = np.array(case["qd"])
        tau = np.array(case["tau"])
        
        print(f"\nCase {i+1}: {case['description']}")
        print(f"  Position: {np.degrees(q[0]):.1f}°")
        print(f"  Velocity: {qd[0]:.2f} rad/s")
        print(f"  Applied torque: {tau[0]:.2f} N⋅m")
        
        # Compute forward dynamics
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        print(f"  → Acceleration: {qdd[0]:.3f} rad/s²")
        
        # Break down the components
        M = pin.compute_mass_matrix(robot, q)
        C_forces = pin.compute_coriolis_forces(robot, q, qd)
        g_forces = pin.compute_gravity_vector(robot, q)
        
        print(f"  Components:")
        print(f"    Mass matrix: {M[0,0]:.3f}")
        print(f"    Coriolis forces: {C_forces[0]:.3f} N⋅m")
        print(f"    Gravity forces: {g_forces[0]:.3f} N⋅m")
        print(f"    Net force: {tau[0] - C_forces[0] - g_forces[0]:.3f} N⋅m")


def simulate_pendulum_motion(robot):
    """Simulate pendulum motion using forward dynamics."""
    print("\n=== Pendulum Simulation ===")
    
    # Initial conditions
    q0 = np.array([np.pi/3])  # Start at 60 degrees
    qd0 = np.array([0.0])     # Start from rest
    
    # Simulation parameters
    dt = 0.01  # Time step
    t_final = 2.0  # Simulation time
    steps = int(t_final / dt)
    
    # Storage for results
    time = np.linspace(0, t_final, steps)
    positions = np.zeros(steps)
    velocities = np.zeros(steps)
    accelerations = np.zeros(steps)
    
    # Initial state
    q = q0.copy()
    qd = qd0.copy()
    
    print(f"Simulating pendulum for {t_final}s with dt={dt}s")
    print(f"Initial position: {np.degrees(q[0]):.1f}°")
    
    for i in range(steps):
        # Store current state
        positions[i] = q[0]
        velocities[i] = qd[0]
        
        # No applied torque - free pendulum motion
        tau = np.array([0.0])
        
        # Compute acceleration using forward dynamics
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        accelerations[i] = qdd[0]
        
        # Integrate using simple Euler method
        # (In practice, use more sophisticated integrators like RK4)
        qd = qd + qdd * dt
        q = q + qd * dt
    
    # Print some key results
    print(f"Final position: {np.degrees(positions[-1]):.1f}°")
    print(f"Maximum velocity: {np.max(np.abs(velocities)):.2f} rad/s")
    print(f"Maximum acceleration: {np.max(np.abs(accelerations)):.2f} rad/s²")
    
    # Energy analysis
    print(f"\nEnergy Analysis:")
    initial_height = -0.5 * np.cos(positions[0])  # Height of COM
    final_height = -0.5 * np.cos(positions[-1])
    
    initial_pe = 1.0 * 9.81 * initial_height  # Potential energy
    final_pe = 1.0 * 9.81 * final_height
    final_ke = 0.5 * 0.083 * velocities[-1]**2  # Kinetic energy (rotational)
    
    print(f"  Initial potential energy: {initial_pe:.3f} J")
    print(f"  Final potential energy: {final_pe:.3f} J")
    print(f"  Final kinetic energy: {final_ke:.3f} J")
    print(f"  Total final energy: {final_pe + final_ke:.3f} J")
    print(f"  Energy conservation error: {abs(initial_pe - (final_pe + final_ke)):.3f} J")


def demonstrate_control_application(robot):
    """Demonstrate using forward dynamics for control."""
    print("\n=== Control Application ===")
    
    # Simple PD controller to stabilize pendulum at upright position
    target_position = 0.0  # Upright (0 degrees)
    
    # Controller gains
    kp = 50.0  # Proportional gain
    kd = 10.0  # Derivative gain
    
    # Initial conditions (start from 30 degrees)
    q = np.array([np.pi/6])
    qd = np.array([0.0])
    
    print(f"PD Controller: Kp={kp}, Kd={kd}")
    print(f"Target: {np.degrees(target_position):.1f}°")
    print(f"Initial position: {np.degrees(q[0]):.1f}°")
    
    # Simulate control
    dt = 0.01
    steps = 200  # 2 seconds
    
    for i in range(steps):
        # PD control law
        error = target_position - q[0]
        error_dot = 0.0 - qd[0]  # Target velocity is 0
        
        tau = np.array([kp * error + kd * error_dot])
        
        # Apply control torque and compute response
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        
        # Integrate
        qd = qd + qdd * dt
        q = q + qd * dt
        
        # Print progress every 0.5 seconds
        if i % 50 == 0:
            print(f"  t={i*dt:.1f}s: pos={np.degrees(q[0]):6.1f}°, "
                  f"vel={qd[0]:6.2f} rad/s, torque={tau[0]:6.2f} N⋅m")
    
    print(f"Final position: {np.degrees(q[0]):.1f}° (error: {np.degrees(abs(q[0])):.1f}°)")


def compare_dynamics_algorithms(robot):
    """Compare different approaches to dynamics computation."""
    print("\n=== Dynamics Algorithm Comparison ===")
    
    # Test configuration
    q = np.array([np.pi/4])
    qd = np.array([0.5])
    tau = np.array([1.0])
    
    print(f"Test configuration:")
    print(f"  Position: {np.degrees(q[0]):.1f}°")
    print(f"  Velocity: {qd[0]:.2f} rad/s")
    print(f"  Torque: {tau[0]:.2f} N⋅m")
    
    # Method 1: Direct matrix inversion
    M = pin.compute_mass_matrix(robot, q)
    C = pin.compute_coriolis_forces(robot, q, qd)
    g = pin.compute_gravity_vector(robot, q)
    
    qdd_direct = np.linalg.solve(M, tau - C - g)
    
    # Method 2: Using forward dynamics function
    qdd_function = pin.compute_forward_dynamics(robot, q, qd, tau)
    
    print(f"\nResults:")
    print(f"  Direct matrix method: {qdd_direct[0]:.6f} rad/s²")
    print(f"  Forward dynamics function: {qdd_function[0]:.6f} rad/s²")
    print(f"  Difference: {abs(qdd_direct[0] - qdd_function[0]):.2e} rad/s²")
    
    # Show component breakdown
    print(f"\nComponent analysis:")
    print(f"  Mass matrix M: {M[0,0]:.3f}")
    print(f"  Coriolis forces C: {C[0]:.3f} N⋅m")
    print(f"  Gravity forces g: {g[0]:.3f} N⋅m")
    print(f"  Net force (τ-C-g): {tau[0] - C[0] - g[0]:.3f} N⋅m")
    print(f"  Acceleration (F/M): {(tau[0] - C[0] - g[0])/M[0,0]:.3f} rad/s²")


def main():
    """Main demonstration function."""
    print("py-pinocchio Forward Dynamics Example")
    print("=" * 50)
    
    # Create robot
    robot = create_pendulum_robot()
    
    # Demonstrate various aspects of forward dynamics
    demonstrate_forward_dynamics_basics(robot)
    simulate_pendulum_motion(robot)
    demonstrate_control_application(robot)
    compare_dynamics_algorithms(robot)
    
    print("\n" + "=" * 50)
    print("Forward dynamics example completed!")
    print("\nKey takeaways:")
    print("- Forward dynamics computes accelerations from torques")
    print("- Essential for robot simulation and control")
    print("- Includes gravitational, Coriolis, and inertial effects")
    print("- Can be used for control system design and validation")


if __name__ == "__main__":
    main()
