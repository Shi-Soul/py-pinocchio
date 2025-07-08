# Performance Guide for py-pinocchio

This guide covers performance optimization, benchmarking, and best practices for efficient robotics computations with py-pinocchio.

## Table of Contents

1. [Performance Philosophy](#performance-philosophy)
2. [Benchmarking Tools](#benchmarking-tools)
3. [Algorithm Complexity](#algorithm-complexity)
4. [Optimization Strategies](#optimization-strategies)
5. [Memory Management](#memory-management)
6. [Numerical Stability](#numerical-stability)

## Performance Philosophy

py-pinocchio prioritizes **educational clarity over raw performance**. However, understanding performance characteristics is crucial for:

- **Algorithm Development**: Choosing appropriate methods for different scenarios
- **Educational Purposes**: Understanding computational complexity
- **Prototyping**: Identifying bottlenecks before production implementation
- **Comparison**: Benchmarking against other implementations

### Design Trade-offs

| Aspect | py-pinocchio Choice | Performance Impact |
|--------|-------------------|-------------------|
| **Code Clarity** | Explicit, readable algorithms | Slower than optimized C++ |
| **Type Safety** | Full type annotations | Runtime overhead |
| **Immutability** | Immutable data structures | Memory allocation overhead |
| **Validation** | Extensive error checking | Additional computation |
| **Modularity** | Pure functions | Function call overhead |

## Benchmarking Tools

### Basic Timing

```python
import time
import numpy as np
import py_pinocchio as pin

def benchmark_function(func, *args, num_runs=1000, warmup=10):
    """
    Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    
    Returns:
        Dictionary with timing statistics
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args)
    
    # Timing runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'total_runs': num_runs
    }

# Example usage
def create_benchmark_robot(num_dof=6):
    """Create robot for benchmarking."""
    links = []
    joints = []
    
    # Base
    base = pin.Link("base", mass=5.0)
    links.append(base)
    
    # Chain of links
    for i in range(num_dof):
        link = pin.Link(
            name=f"link{i+1}",
            mass=2.0 - i*0.1,
            center_of_mass=np.array([0.1, 0.0, 0.0]),
            inertia_tensor=np.diag([0.01, 0.01, 0.005])
        )
        links.append(link)
        
        parent = "base" if i == 0 else f"link{i}"
        joint = pin.Joint(
            name=f"joint{i+1}",
            joint_type=pin.JointType.REVOLUTE,
            axis=np.array([0, 0, 1]) if i % 2 == 0 else np.array([0, 1, 0]),
            parent_link=parent,
            child_link=f"link{i+1}",
            origin_transform=pin.create_transform(np.eye(3), np.array([0.2, 0, 0]))
        )
        joints.append(joint)
    
    return pin.create_robot_model(f"benchmark_robot_{num_dof}dof", links, joints, "base")

# Benchmark forward kinematics
robot = create_benchmark_robot(6)
q = np.random.uniform(-np.pi, np.pi, robot.num_dof)

fk_stats = benchmark_function(pin.compute_forward_kinematics, robot, q)
print("Forward Kinematics Benchmark:")
print(f"  Mean time: {fk_stats['mean_time']*1000:.3f} ms")
print(f"  Std dev: {fk_stats['std_time']*1000:.3f} ms")
print(f"  Min time: {fk_stats['min_time']*1000:.3f} ms")
```

### Comprehensive Benchmarking Suite

```python
def comprehensive_benchmark(robot_sizes=[2, 4, 6, 8, 10], num_runs=100):
    """
    Comprehensive benchmarking across different robot sizes.
    
    Args:
        robot_sizes: List of DOF counts to test
        num_runs: Number of runs per test
    
    Returns:
        Benchmark results dictionary
    """
    results = {}
    
    for num_dof in robot_sizes:
        print(f"\nBenchmarking {num_dof}-DOF robot...")
        
        # Create robot
        robot = create_benchmark_robot(num_dof)
        q = np.random.uniform(-np.pi, np.pi, num_dof)
        qd = np.random.uniform(-1, 1, num_dof)
        qdd = np.random.uniform(-1, 1, num_dof)
        tau = np.random.uniform(-10, 10, num_dof)
        
        robot_results = {}
        
        # Forward kinematics
        fk_stats = benchmark_function(
            pin.compute_forward_kinematics, robot, q, num_runs=num_runs
        )
        robot_results['forward_kinematics'] = fk_stats
        
        # Jacobian computation
        jacobian_stats = benchmark_function(
            pin.compute_geometric_jacobian, robot, q, robot.links[-1].name, num_runs=num_runs
        )
        robot_results['jacobian'] = jacobian_stats
        
        # Mass matrix
        mass_matrix_stats = benchmark_function(
            pin.compute_mass_matrix, robot, q, num_runs=num_runs
        )
        robot_results['mass_matrix'] = mass_matrix_stats
        
        # Gravity vector
        gravity_stats = benchmark_function(
            pin.compute_gravity_vector, robot, q, num_runs=num_runs
        )
        robot_results['gravity'] = gravity_stats
        
        # Forward dynamics
        forward_dynamics_stats = benchmark_function(
            pin.compute_forward_dynamics, robot, q, qd, tau, num_runs=num_runs
        )
        robot_results['forward_dynamics'] = forward_dynamics_stats
        
        # Inverse dynamics
        inverse_dynamics_stats = benchmark_function(
            pin.compute_inverse_dynamics, robot, q, qd, qdd, num_runs=num_runs
        )
        robot_results['inverse_dynamics'] = inverse_dynamics_stats
        
        results[num_dof] = robot_results
        
        # Print summary
        print(f"  Forward Kinematics: {fk_stats['mean_time']*1000:.2f} ms")
        print(f"  Jacobian: {jacobian_stats['mean_time']*1000:.2f} ms")
        print(f"  Mass Matrix: {mass_matrix_stats['mean_time']*1000:.2f} ms")
        print(f"  Forward Dynamics: {forward_dynamics_stats['mean_time']*1000:.2f} ms")
        print(f"  Inverse Dynamics: {inverse_dynamics_stats['mean_time']*1000:.2f} ms")
    
    return results

# Run comprehensive benchmark
benchmark_results = comprehensive_benchmark()
```

### Scaling Analysis

```python
def analyze_scaling(benchmark_results):
    """
    Analyze computational scaling with robot size.
    
    Args:
        benchmark_results: Results from comprehensive_benchmark
    """
    import matplotlib.pyplot as plt
    
    dof_counts = sorted(benchmark_results.keys())
    algorithms = ['forward_kinematics', 'jacobian', 'mass_matrix', 
                 'forward_dynamics', 'inverse_dynamics']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, algorithm in enumerate(algorithms):
        ax = axes[i]
        
        times = [benchmark_results[dof][algorithm]['mean_time'] * 1000 
                for dof in dof_counts]
        
        ax.loglog(dof_counts, times, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Degrees of Freedom')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{algorithm.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Fit scaling law
        log_dof = np.log(dof_counts)
        log_times = np.log(times)
        coeffs = np.polyfit(log_dof, log_times, 1)
        scaling_exponent = coeffs[0]
        
        # Plot fitted line
        fitted_times = np.exp(coeffs[1]) * np.array(dof_counts) ** scaling_exponent
        ax.loglog(dof_counts, fitted_times, '--', alpha=0.7, 
                 label=f'O(n^{scaling_exponent:.1f})')
        ax.legend()
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print scaling summary
    print("\nScaling Analysis Summary:")
    print("=" * 40)
    
    for algorithm in algorithms:
        times = [benchmark_results[dof][algorithm]['mean_time'] * 1000 
                for dof in dof_counts]
        
        log_dof = np.log(dof_counts)
        log_times = np.log(times)
        coeffs = np.polyfit(log_dof, log_times, 1)
        scaling_exponent = coeffs[0]
        
        print(f"{algorithm.replace('_', ' ').title()}: O(n^{scaling_exponent:.1f})")

# Analyze scaling
analyze_scaling(benchmark_results)
```

## Algorithm Complexity

### Theoretical Complexity

| Algorithm | Complexity | py-pinocchio Implementation |
|-----------|------------|---------------------------|
| **Forward Kinematics** | $O(n)$ | $O(n)$ - Linear chain traversal |
| **Jacobian** | $O(n)$ | $O(n)$ - Single pass computation |
| **Mass Matrix** | $O(n^2)$ | $O(n^2)$ - CRBA algorithm |
| **Inverse Dynamics** | $O(n)$ | $O(n)$ - RNEA algorithm |
| **Forward Dynamics** | $O(n^3)$ | $O(n^3)$ - Matrix inversion |

### Memory Complexity

| Data Structure | Memory Usage | Scaling |
|----------------|--------------|---------|
| **Robot Model** | $O(n)$ | Links and joints |
| **Joint Configuration** | $O(n)$ | Joint angles |
| **Mass Matrix** | $O(n^2)$ | Dense matrix |
| **Jacobian** | $O(n)$ | 6×n matrix |
| **Kinematic State** | $O(n)$ | Transform per link |

## Optimization Strategies

### 1. Algorithm Selection

```python
def choose_optimal_algorithm(robot, use_case):
    """
    Choose optimal algorithm based on use case.
    
    Args:
        robot: Robot model
        use_case: 'control', 'simulation', 'analysis', 'real_time'
    
    Returns:
        Recommended algorithms and settings
    """
    recommendations = {}
    
    if use_case == 'real_time':
        recommendations.update({
            'dynamics': 'inverse_dynamics',  # O(n) vs O(n³)
            'jacobian': 'geometric',         # Faster than analytical
            'mass_matrix': 'avoid_if_possible',  # O(n²) computation
            'caching': 'aggressive'
        })
    
    elif use_case == 'control':
        recommendations.update({
            'dynamics': 'both',              # Need both forward and inverse
            'jacobian': 'geometric',         # For velocity control
            'mass_matrix': 'compute_once',   # For inertia compensation
            'caching': 'moderate'
        })
    
    elif use_case == 'simulation':
        recommendations.update({
            'dynamics': 'forward_dynamics',  # Integrate accelerations
            'jacobian': 'as_needed',         # For constraints
            'mass_matrix': 'efficient_solver',  # Use sparse methods
            'caching': 'minimal'
        })
    
    elif use_case == 'analysis':
        recommendations.update({
            'dynamics': 'all',               # Complete analysis
            'jacobian': 'both_types',        # Geometric and analytical
            'mass_matrix': 'full_computation',  # All properties
            'caching': 'extensive'
        })
    
    return recommendations

# Example usage
robot = create_benchmark_robot(6)
control_recommendations = choose_optimal_algorithm(robot, 'control')
print("Control recommendations:", control_recommendations)
```

### 2. Caching Strategies

```python
class CachedRobotComputation:
    """
    Cached computation wrapper for improved performance.
    """
    
    def __init__(self, robot):
        self.robot = robot
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _cache_key(self, func_name, *args):
        """Generate cache key from function name and arguments."""
        # Convert numpy arrays to tuples for hashing
        hashable_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                hashable_args.append(tuple(arg.flatten()))
            else:
                hashable_args.append(arg)
        return (func_name, tuple(hashable_args))
    
    def compute_forward_kinematics(self, q):
        """Cached forward kinematics."""
        key = self._cache_key('forward_kinematics', q)
        
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        
        self._cache_misses += 1
        result = pin.compute_forward_kinematics(self.robot, q)
        self._cache[key] = result
        return result
    
    def compute_mass_matrix(self, q):
        """Cached mass matrix computation."""
        key = self._cache_key('mass_matrix', q)
        
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        
        self._cache_misses += 1
        result = pin.compute_mass_matrix(self.robot, q)
        self._cache[key] = result
        return result
    
    def clear_cache(self):
        """Clear computation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def cache_statistics(self):
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }

# Example usage
robot = create_benchmark_robot(4)
cached_robot = CachedRobotComputation(robot)

# Benchmark with and without caching
q = np.array([0.1, 0.2, 0.3, 0.4])

# Without caching
no_cache_stats = benchmark_function(pin.compute_mass_matrix, robot, q, num_runs=100)

# With caching (repeated computations)
def cached_computation():
    return cached_robot.compute_mass_matrix(q)

cache_stats = benchmark_function(cached_computation, num_runs=100)

print("Performance Comparison:")
print(f"No cache: {no_cache_stats['mean_time']*1000:.3f} ms")
print(f"With cache: {cache_stats['mean_time']*1000:.3f} ms")
print(f"Speedup: {no_cache_stats['mean_time']/cache_stats['mean_time']:.1f}x")
print(f"Cache statistics: {cached_robot.cache_statistics()}")
```

### 3. Vectorization

```python
def vectorized_forward_kinematics(robot, q_batch):
    """
    Compute forward kinematics for batch of configurations.
    
    Args:
        robot: Robot model
        q_batch: Array of shape (batch_size, num_dof)
    
    Returns:
        Batch of kinematic states
    """
    batch_size = q_batch.shape[0]
    results = []
    
    for i in range(batch_size):
        result = pin.compute_forward_kinematics(robot, q_batch[i])
        results.append(result)
    
    return results

def benchmark_vectorization():
    """Compare single vs batch computation."""
    robot = create_benchmark_robot(6)
    batch_size = 100
    
    # Generate batch of configurations
    q_batch = np.random.uniform(-np.pi, np.pi, (batch_size, robot.num_dof))
    
    # Single computation timing
    def single_computations():
        for q in q_batch:
            pin.compute_forward_kinematics(robot, q)
    
    # Batch computation timing
    def batch_computation():
        vectorized_forward_kinematics(robot, q_batch)
    
    single_stats = benchmark_function(single_computations, num_runs=10)
    batch_stats = benchmark_function(batch_computation, num_runs=10)
    
    print("Vectorization Benchmark:")
    print(f"Single computations: {single_stats['mean_time']*1000:.1f} ms")
    print(f"Batch computation: {batch_stats['mean_time']*1000:.1f} ms")
    print(f"Efficiency: {single_stats['mean_time']/batch_stats['mean_time']:.1f}x")

benchmark_vectorization()
```

## Memory Management

### Memory Profiling

```python
import tracemalloc

def profile_memory_usage(func, *args):
    """
    Profile memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Function arguments
    
    Returns:
        Memory usage statistics
    """
    tracemalloc.start()
    
    # Take snapshot before
    snapshot_before = tracemalloc.take_snapshot()
    
    # Execute function
    result = func(*args)
    
    # Take snapshot after
    snapshot_after = tracemalloc.take_snapshot()
    
    # Calculate difference
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    total_memory = sum(stat.size for stat in top_stats)
    
    tracemalloc.stop()
    
    return {
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / (1024 * 1024),
        'top_allocations': top_stats[:5]
    }

# Profile memory usage
robot = create_benchmark_robot(8)
q = np.random.uniform(-np.pi, np.pi, robot.num_dof)

memory_stats = profile_memory_usage(pin.compute_mass_matrix, robot, q)
print(f"Memory usage: {memory_stats['total_memory_mb']:.2f} MB")
```

### Memory-Efficient Patterns

```python
def memory_efficient_simulation(robot, trajectory, dt=0.01):
    """
    Memory-efficient simulation using in-place operations.
    
    Args:
        robot: Robot model
        trajectory: Desired trajectory
        dt: Time step
    """
    # Pre-allocate arrays
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    qdd = np.zeros(robot.num_dof)
    tau = np.zeros(robot.num_dof)
    
    # Reuse arrays instead of creating new ones
    for i, q_desired in enumerate(trajectory):
        # Update configuration (in-place)
        q[:] = q_desired
        
        # Compute dynamics (reusing arrays)
        M = pin.compute_mass_matrix(robot, q)
        g = pin.compute_gravity_vector(robot, q)
        
        # Simple PD control
        tau[:] = 100 * (q_desired - q) - 10 * qd + g
        
        # Forward dynamics
        qdd[:] = pin.compute_forward_dynamics(robot, q, qd, tau)
        
        # Integration (in-place)
        qd += qdd * dt
        q += qd * dt
    
    return q, qd

# Compare with memory-inefficient version
def memory_inefficient_simulation(robot, trajectory, dt=0.01):
    """Memory-inefficient version creating new arrays."""
    q = np.zeros(robot.num_dof)
    qd = np.zeros(robot.num_dof)
    
    for q_desired in trajectory:
        M = pin.compute_mass_matrix(robot, q)
        g = pin.compute_gravity_vector(robot, q)
        
        # Creates new arrays each iteration
        tau = 100 * (q_desired - q) - 10 * qd + g
        qdd = pin.compute_forward_dynamics(robot, q, qd, tau)
        
        # Creates new arrays
        qd = qd + qdd * dt
        q = q + qd * dt
    
    return q, qd
```

## Numerical Stability

### Condition Number Monitoring

```python
def monitor_numerical_stability(robot, q_samples):
    """
    Monitor numerical stability across workspace.
    
    Args:
        robot: Robot model
        q_samples: Array of joint configurations to test
    
    Returns:
        Stability analysis results
    """
    condition_numbers = []
    singular_configs = []
    
    for i, q in enumerate(q_samples):
        try:
            # Compute Jacobian
            J = pin.compute_geometric_jacobian(robot, q, robot.links[-1].name)
            
            # Condition number
            cond_num = np.linalg.cond(J)
            condition_numbers.append(cond_num)
            
            # Check for near-singularity
            if cond_num > 1000:
                singular_configs.append((i, q, cond_num))
            
            # Mass matrix condition
            M = pin.compute_mass_matrix(robot, q)
            M_cond = np.linalg.cond(M)
            
            if M_cond > 1e12:
                print(f"Warning: Ill-conditioned mass matrix at config {i}")
        
        except Exception as e:
            print(f"Numerical error at config {i}: {e}")
    
    condition_numbers = np.array(condition_numbers)
    
    return {
        'mean_condition': np.mean(condition_numbers),
        'max_condition': np.max(condition_numbers),
        'singular_configs': singular_configs,
        'stability_ratio': len(singular_configs) / len(q_samples)
    }

# Test numerical stability
robot = create_benchmark_robot(6)
test_configs = [np.random.uniform(-np.pi, np.pi, robot.num_dof) for _ in range(100)]

stability_results = monitor_numerical_stability(robot, test_configs)
print("Numerical Stability Analysis:")
print(f"Mean condition number: {stability_results['mean_condition']:.2f}")
print(f"Max condition number: {stability_results['max_condition']:.2f}")
print(f"Singular configurations: {len(stability_results['singular_configs'])}")
print(f"Stability ratio: {stability_results['stability_ratio']:.2%}")
```

This performance guide provides comprehensive tools and strategies for optimizing py-pinocchio computations while maintaining the library's educational focus and code clarity.
