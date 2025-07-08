#!/usr/bin/env python3
"""
Comprehensive test runner for py-pinocchio

This script runs all test suites and provides detailed reporting.
It includes tests for:
- Basic functionality
- Legged robots (bipeds, quadrupeds)
- Multi-DOF arms (planar, spatial, redundant)
- Edge cases and numerical stability
- Performance benchmarks
"""

import sys
import os
import time
import traceback
import numpy as np
from typing import List, Tuple, Dict, Any

# Add parent directory to path to import py_pinocchio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import py_pinocchio as pin


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.duration = 0.0
        self.error_message = ""
        self.details = {}


class TestRunner:
    """Comprehensive test runner with reporting."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_start_time = 0.0
        
    def run_test_suite(self, test_function, suite_name: str) -> TestResult:
        """Run a test suite and capture results."""
        result = TestResult(suite_name)
        
        print(f"\n{'='*60}")
        print(f"Running {suite_name}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            test_function()
            result.passed = True
            print(f"✓ {suite_name} passed")
            
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            print(f"✗ {suite_name} failed: {e}")
            print("Traceback:")
            traceback.print_exc()
        
        result.duration = time.time() - start_time
        self.results.append(result)
        
        return result
    
    def run_all_tests(self):
        """Run all available test suites."""
        self.total_start_time = time.time()
        
        print("py-pinocchio Comprehensive Test Suite")
        print("="*60)
        print(f"Python version: {sys.version}")
        print(f"py-pinocchio version: {pin.__version__}")
        print(f"Test directory: {os.path.dirname(__file__)}")
        
        # Test suites to run
        test_suites = [
            ("Basic Functionality", self._run_basic_tests),
            ("Legged Robots", self._run_legged_robot_tests),
            ("Multi-DOF Arms", self._run_multi_dof_arm_tests),
            ("Edge Cases", self._run_edge_case_tests),
            ("Performance Benchmarks", self._run_performance_tests),
        ]
        
        # Run each test suite
        for suite_name, test_function in test_suites:
            self.run_test_suite(test_function, suite_name)
        
        # Generate final report
        self._generate_report()
    
    def _run_basic_tests(self):
        """Run basic functionality tests."""
        try:
            from test_basic_functionality import run_basic_tests
            run_basic_tests()
        except ImportError:
            # Run inline basic tests if module not available
            self._run_inline_basic_tests()
    
    def _run_legged_robot_tests(self):
        """Run legged robot tests."""
        from test_legged_robots import run_legged_robot_tests
        run_legged_robot_tests()
    
    def _run_multi_dof_arm_tests(self):
        """Run multi-DOF arm tests."""
        from test_multi_dof_arms import run_multi_dof_arm_tests
        run_multi_dof_arm_tests()
    
    def _run_edge_case_tests(self):
        """Run edge case tests."""
        from test_edge_cases import run_edge_case_tests
        run_edge_case_tests()
    
    def _run_performance_tests(self):
        """Run performance benchmark tests."""
        print("Running performance benchmarks...")
        
        # Create test robot for benchmarking
        robot = self._create_benchmark_robot()
        
        # Benchmark forward kinematics
        self._benchmark_forward_kinematics(robot)
        
        # Benchmark dynamics
        self._benchmark_dynamics(robot)
        
        # Benchmark Jacobian computation
        self._benchmark_jacobian(robot)
    
    def _create_benchmark_robot(self):
        """Create a robot for performance benchmarking."""
        # Create 6-DOF arm for benchmarking
        links = []
        joints = []
        
        # Base
        base = pin.Link("base", mass=5.0)
        links.append(base)
        
        # Create 6 links and joints
        for i in range(6):
            link = pin.Link(
                name=f"link{i+1}",
                mass=2.0 - i*0.2,
                center_of_mass=np.array([0.2, 0.0, 0.0]),
                inertia_tensor=np.diag([0.1, 0.1, 0.05])
            )
            links.append(link)
            
            parent = "base" if i == 0 else f"link{i}"
            axis = [0, 0, 1] if i % 2 == 0 else [0, 1, 0]
            offset = [0, 0, 0.1] if i == 0 else [0.2, 0, 0]
            
            joint = pin.Joint(
                name=f"joint{i+1}",
                joint_type=pin.JointType.REVOLUTE,
                axis=np.array(axis),
                parent_link=parent,
                child_link=f"link{i+1}",
                origin_transform=pin.create_transform(np.eye(3), np.array(offset))
            )
            joints.append(joint)
        
        return pin.create_robot_model(
            name="benchmark_robot",
            links=links,
            joints=joints,
            root_link="base"
        )
    
    def _benchmark_forward_kinematics(self, robot):
        """Benchmark forward kinematics computation."""
        import numpy as np
        
        print("Benchmarking forward kinematics...")
        
        n_iterations = 1000
        q = np.random.uniform(-np.pi, np.pi, robot.num_dof)
        
        start_time = time.time()
        for _ in range(n_iterations):
            kinematic_state = pin.compute_forward_kinematics(robot, q)
        duration = time.time() - start_time
        
        avg_time = duration / n_iterations * 1000  # ms
        print(f"  Forward kinematics: {avg_time:.3f} ms/call ({n_iterations} iterations)")
    
    def _benchmark_dynamics(self, robot):
        """Benchmark dynamics computation."""
        import numpy as np
        
        print("Benchmarking dynamics...")
        
        n_iterations = 100
        q = np.random.uniform(-np.pi, np.pi, robot.num_dof)
        qd = np.random.uniform(-1, 1, robot.num_dof)
        tau = np.random.uniform(-10, 10, robot.num_dof)
        
        # Benchmark mass matrix
        start_time = time.time()
        for _ in range(n_iterations):
            M = pin.compute_mass_matrix(robot, q)
        duration = time.time() - start_time
        avg_time = duration / n_iterations * 1000
        print(f"  Mass matrix: {avg_time:.3f} ms/call")
        
        # Benchmark forward dynamics
        start_time = time.time()
        for _ in range(n_iterations):
            qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        duration = time.time() - start_time
        avg_time = duration / n_iterations * 1000
        print(f"  Forward dynamics: {avg_time:.3f} ms/call")
    
    def _benchmark_jacobian(self, robot):
        """Benchmark Jacobian computation."""
        import numpy as np
        
        print("Benchmarking Jacobian...")
        
        n_iterations = 500
        q = np.random.uniform(-np.pi, np.pi, robot.num_dof)
        
        start_time = time.time()
        for _ in range(n_iterations):
            J = pin.compute_geometric_jacobian(robot, q, f"link{robot.num_dof}")
        duration = time.time() - start_time
        
        avg_time = duration / n_iterations * 1000
        print(f"  Geometric Jacobian: {avg_time:.3f} ms/call")
    
    def _run_inline_basic_tests(self):
        """Run basic tests inline if module not available."""
        import numpy as np
        
        print("Running inline basic tests...")
        
        # Test transform creation
        T = pin.create_identity_transform()
        assert np.allclose(T.rotation, np.eye(3))
        assert np.allclose(T.translation, np.zeros(3))
        
        # Test simple robot creation
        base = pin.Link("base", mass=1.0)
        link1 = pin.Link("link1", mass=0.5, center_of_mass=np.array([0.2, 0, 0]))
        
        joint1 = pin.Joint(
            name="joint1",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]),
            parent_link="base",
            child_link="link1"
        )
        
        robot = pin.create_robot_model(
            name="test_robot",
            links=[base, link1],
            joints=[joint1],
            root_link="base"
        )
        
        # Test forward kinematics
        q = np.array([np.pi/4])
        ee_pos = pin.get_link_position(robot, q, "link1")
        assert len(ee_pos) == 3
        
        print("✓ Inline basic tests passed")
    
    def _generate_report(self):
        """Generate final test report."""
        total_duration = time.time() - self.total_start_time
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        print(f"Total test suites: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Total duration: {total_duration:.2f} seconds")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{result.name:<25} {status:<8} {result.duration:.2f}s")
            
            if not result.passed:
                print(f"  Error: {result.error_message}")
        
        if passed_count == total_count:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total_count - passed_count} test suite(s) failed")
            sys.exit(1)


def main():
    """Main entry point."""
    runner = TestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
