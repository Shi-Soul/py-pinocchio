#!/usr/bin/env python3
"""
Simple test of forward dynamics functionality
"""

import numpy as np
import py_pinocchio as pin

def test_forward_dynamics():
    """Test forward dynamics with a simple robot."""
    print("Testing Forward Dynamics Implementation")
    print("=" * 40)
    
    # Create simple 2-DOF robot
    base = pin.Link("base", mass=0.0)
    link1 = pin.Link("link1", mass=1.0, center_of_mass=[0.5, 0, 0])
    link2 = pin.Link("link2", mass=0.5, center_of_mass=[0.3, 0, 0])
    
    joint1 = pin.Joint(
        name="joint1",
        joint_type=pin.JointType.REVOLUTE,
        axis=[0, 0, 1],
        parent_link="base",
        child_link="link1",
        position_limit_lower=-10*np.pi,
        position_limit_upper=10*np.pi
    )
    
    joint2_origin = pin.create_transform(np.eye(3), [1, 0, 0])
    joint2 = pin.Joint(
        name="joint2",
        joint_type=pin.JointType.REVOLUTE,
        axis=[0, 0, 1],
        parent_link="link1",
        child_link="link2",
        origin_transform=joint2_origin,
        position_limit_lower=-10*np.pi,
        position_limit_upper=10*np.pi
    )
    
    robot = pin.create_robot_model("test_robot", [base, link1, link2], [joint1, joint2], "base")
    
    print(f"Created robot with {robot.num_dof} DOF")
    
    # Test configurations
    test_cases = [
        {"q": [0.0, 0.0], "qd": [0.0, 0.0], "tau": [0.0, 0.0], "desc": "Zero config, no torque"},
        {"q": [0.0, 0.0], "qd": [0.0, 0.0], "tau": [1.0, 0.0], "desc": "Zero config, torque on joint 1"},
        {"q": [0.0, 0.0], "qd": [0.0, 0.0], "tau": [0.0, 1.0], "desc": "Zero config, torque on joint 2"},
        {"q": [0.5, 0.3], "qd": [0.1, -0.2], "tau": [0.5, -0.3], "desc": "General configuration"},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['desc']}")
        q = np.array(case["q"])
        qd = np.array(case["qd"])
        tau = np.array(case["tau"])
        
        print(f"  q = {q}")
        print(f"  qd = {qd}")
        print(f"  tau = {tau}")
        
        try:
            # Compute forward dynamics
            qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
            print(f"  → qdd = {qdd}")
            
            # Verify with components
            M = pin.compute_mass_matrix(robot, q)
            C = pin.compute_coriolis_forces(robot, q, qd)
            g = pin.compute_gravity_vector(robot, q)
            
            print(f"  Mass matrix diagonal: {np.diag(M)}")
            print(f"  Coriolis forces: {C}")
            print(f"  Gravity forces: {g}")
            
            # Manual verification: qdd = M^-1 * (tau - C - g)
            qdd_manual = np.linalg.solve(M, tau - C - g)
            print(f"  Manual calculation: {qdd_manual}")
            print(f"  Difference: {np.abs(qdd - qdd_manual)}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "=" * 40)
    print("Forward dynamics test completed!")

if __name__ == "__main__":
    test_forward_dynamics()
