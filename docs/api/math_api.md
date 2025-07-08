# Mathematical Utilities API Reference

This document provides comprehensive API documentation for py-pinocchio's mathematical utilities, including transformations, spatial algebra, and numerical methods.

## Table of Contents

1. [Transformations](#transformations)
2. [Spatial Algebra](#spatial-algebra)
3. [Rotation Utilities](#rotation-utilities)
4. [Numerical Methods](#numerical-methods)
5. [Linear Algebra](#linear-algebra)

## Transformations

### Transform Class

```python
class Transform(NamedTuple):
    """Immutable 3D transformation representation."""
    rotation: Matrix3x3
    translation: Vector3
```

#### Properties

- **`rotation`** (`Matrix3x3`): $3 \times 3$ rotation matrix $\mathbf{R} \in SO(3)$
- **`translation`** (`Vector3`): 3D translation vector $\mathbf{t} \in \mathbb{R}^3$

#### Mathematical Representation

A transformation represents the homogeneous transformation:

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)$$

### Core Transform Functions

#### `create_identity_transform() -> Transform`

Create identity transformation.

**Returns:**
- Identity transform: $\mathbf{T} = \mathbf{I}_4$

**Example:**
```python
T_identity = pin.create_identity_transform()
assert np.allclose(T_identity.rotation, np.eye(3))
assert np.allclose(T_identity.translation, np.zeros(3))
```

#### `create_transform(rotation: Matrix3x3, translation: Vector3) -> Transform`

Create transformation from rotation matrix and translation vector.

**Parameters:**
- `rotation`: $3 \times 3$ rotation matrix (must be orthogonal)
- `translation`: 3D translation vector

**Returns:**
- Transform object

**Raises:**
- `ValueError`: If rotation matrix is not orthogonal

**Mathematical Operation:**
$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**Example:**
```python
R = pin.math.utils.rodrigues_formula(np.array([0, 0, 1]), np.pi/4)
t = np.array([1.0, 2.0, 3.0])
T = pin.create_transform(R, t)
```

#### `compose_transforms(T1: Transform, T2: Transform) -> Transform`

Compose two transformations.

**Parameters:**
- `T1`: First transformation (applied second)
- `T2`: Second transformation (applied first)

**Returns:**
- Composed transformation $\mathbf{T}_1 \mathbf{T}_2$

**Mathematical Operation:**
$$\mathbf{T}_{result} = \mathbf{T}_1 \mathbf{T}_2$$

Point transformation: $\mathbf{p}' = \mathbf{T}_1(\mathbf{T}_2(\mathbf{p}))$

**Example:**
```python
T1 = pin.create_transform(np.eye(3), np.array([1, 0, 0]))
T2 = pin.create_transform(np.eye(3), np.array([0, 1, 0]))
T_composed = pin.compose_transforms(T1, T2)
# Result: translation = [1, 1, 0]
```

#### `invert_transform(transform: Transform) -> Transform`

Compute inverse of transformation.

**Parameters:**
- `transform`: Transform to invert

**Returns:**
- Inverse transformation $\mathbf{T}^{-1}$

**Mathematical Operation:**
$$\mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**Properties:**
- $\mathbf{T} \mathbf{T}^{-1} = \mathbf{I}$
- $(\mathbf{T}_1 \mathbf{T}_2)^{-1} = \mathbf{T}_2^{-1} \mathbf{T}_1^{-1}$

#### `transform_point(transform: Transform, point: Vector3) -> Vector3`

Apply transformation to a 3D point.

**Parameters:**
- `transform`: Transformation to apply
- `point`: 3D point to transform

**Returns:**
- Transformed point

**Mathematical Operation:**
$$\mathbf{p}' = \mathbf{R} \mathbf{p} + \mathbf{t}$$

#### `transform_vector(transform: Transform, vector: Vector3) -> Vector3`

Apply only rotation part of transformation to a vector.

**Parameters:**
- `transform`: Transformation (only rotation used)
- `vector`: 3D vector to transform

**Returns:**
- Transformed vector

**Mathematical Operation:**
$$\mathbf{v}' = \mathbf{R} \mathbf{v}$$

**Note:** Vectors are not affected by translation, only rotation.

## Spatial Algebra

Spatial algebra provides efficient 6D representations for robot dynamics.

### Spatial Vector Operations

#### `create_spatial_vector(angular: Vector3, linear: Vector3) -> Vector6`

Create 6D spatial vector.

**Parameters:**
- `angular`: 3D angular component (e.g., angular velocity)
- `linear`: 3D linear component (e.g., linear velocity)

**Returns:**
- 6D spatial vector

**Mathematical Representation:**
$$\mathbf{v} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} \in \mathbb{R}^6$$

#### `spatial_cross_product(v1: Vector6, v2: Vector6) -> Vector6`

Compute spatial cross product.

**Parameters:**
- `v1`, `v2`: 6D spatial vectors

**Returns:**
- Spatial cross product $\mathbf{v}_1 \times^* \mathbf{v}_2$

**Mathematical Operation:**
$$\mathbf{v}_1 \times^* \mathbf{v}_2 = \begin{bmatrix} \boldsymbol{\omega}_1 \times \boldsymbol{\omega}_2 + \mathbf{v}_1 \times \mathbf{v}_2 \\ \boldsymbol{\omega}_1 \times \mathbf{v}_2 \end{bmatrix}$$

**Properties:**
- Bilinear: $(a\mathbf{v}_1 + b\mathbf{v}_2) \times^* \mathbf{v}_3 = a(\mathbf{v}_1 \times^* \mathbf{v}_3) + b(\mathbf{v}_2 \times^* \mathbf{v}_3)$
- Anti-symmetric: $\mathbf{v}_1 \times^* \mathbf{v}_2 = -(\mathbf{v}_2 \times^* \mathbf{v}_1)$

### Spatial Transformations

#### `create_spatial_transform(transform: Transform) -> Matrix6x6`

Create $6 \times 6$ spatial transformation matrix.

**Parameters:**
- `transform`: 3D transformation

**Returns:**
- $6 \times 6$ spatial transformation matrix

**Mathematical Operation:**
$$\mathbf{X} = \begin{bmatrix} \mathbf{R} & \mathbf{0} \\ \mathbf{r}^\times \mathbf{R} & \mathbf{R} \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

Where $\mathbf{r}^\times$ is the skew-symmetric matrix of translation vector.

#### `spatial_transform_motion(X: Matrix6x6, v: Vector6) -> Vector6`

Transform spatial motion vector.

**Parameters:**
- `X`: $6 \times 6$ spatial transformation matrix
- `v`: 6D spatial motion vector

**Returns:**
- Transformed spatial motion vector

**Mathematical Operation:**
$$\mathbf{v}' = \mathbf{X} \mathbf{v}$$

#### `spatial_transform_force(X: Matrix6x6, f: Vector6) -> Vector6`

Transform spatial force vector.

**Parameters:**
- `X`: $6 \times 6$ spatial transformation matrix
- `f`: 6D spatial force vector

**Returns:**
- Transformed spatial force vector

**Mathematical Operation:**
$$\mathbf{f}' = \mathbf{X}^T \mathbf{f}$$

**Note:** Forces transform with the transpose of the motion transformation.

## Rotation Utilities

### Rotation Representations

#### `rodrigues_formula(axis: Vector3, angle: float) -> Matrix3x3`

Compute rotation matrix using Rodrigues' formula.

**Parameters:**
- `axis`: Unit rotation axis $\mathbf{k} \in S^2$
- `angle`: Rotation angle in radians

**Returns:**
- $3 \times 3$ rotation matrix

**Mathematical Operation:**
$$\mathbf{R} = \mathbf{I} + \sin(\theta)[\mathbf{k}]_\times + (1-\cos(\theta))[\mathbf{k}]_\times^2$$

Where $[\mathbf{k}]_\times$ is the skew-symmetric matrix of $\mathbf{k}$.

#### `rotation_matrix_to_axis_angle(R: Matrix3x3) -> Tuple[Vector3, float]`

Extract axis-angle representation from rotation matrix.

**Parameters:**
- `R`: $3 \times 3$ rotation matrix

**Returns:**
- Tuple of (axis, angle)

**Mathematical Operation:**
$$\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right)$$
$$\mathbf{k} = \frac{1}{2\sin(\theta)} \begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix}$$

#### `skew_symmetric(v: Vector3) -> Matrix3x3`

Create skew-symmetric matrix from 3D vector.

**Parameters:**
- `v`: 3D vector $\mathbf{v} = [v_x, v_y, v_z]^T$

**Returns:**
- $3 \times 3$ skew-symmetric matrix

**Mathematical Operation:**
$$[\mathbf{v}]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}$$

**Properties:**
- $[\mathbf{v}]_\times^T = -[\mathbf{v}]_\times$ (skew-symmetric)
- $[\mathbf{v}]_\times \mathbf{w} = \mathbf{v} \times \mathbf{w}$ (cross product)

#### `unskew_symmetric(S: Matrix3x3) -> Vector3`

Extract vector from skew-symmetric matrix.

**Parameters:**
- `S`: $3 \times 3$ skew-symmetric matrix

**Returns:**
- 3D vector

**Mathematical Operation:**
Inverse of `skew_symmetric`: extracts $\mathbf{v}$ from $[\mathbf{v}]_\times$.

### Euler Angles

#### `rotation_matrix_to_euler_xyz(R: Matrix3x3) -> Vector3`

Convert rotation matrix to X-Y-Z Euler angles.

**Parameters:**
- `R`: $3 \times 3$ rotation matrix

**Returns:**
- Euler angles $[\phi, \theta, \psi]^T$ (roll, pitch, yaw)

**Mathematical Operation:**
$$\mathbf{R} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

#### `euler_xyz_to_rotation_matrix(euler: Vector3) -> Matrix3x3`

Convert X-Y-Z Euler angles to rotation matrix.

**Parameters:**
- `euler`: Euler angles $[\phi, \theta, \psi]^T$

**Returns:**
- $3 \times 3$ rotation matrix

## Numerical Methods

### Matrix Operations

#### `matrix_pseudoinverse(A: np.ndarray, tolerance: float = 1e-10) -> np.ndarray`

Compute Moore-Penrose pseudoinverse.

**Parameters:**
- `A`: Input matrix
- `tolerance`: Singular value threshold

**Returns:**
- Pseudoinverse matrix $\mathbf{A}^+$

**Mathematical Operation:**
$$\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T$$

Where $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ is the SVD.

#### `damped_pseudoinverse(A: np.ndarray, damping: float = 1e-6) -> np.ndarray`

Compute damped pseudoinverse for numerical stability.

**Parameters:**
- `A`: Input matrix
- `damping`: Damping parameter $\lambda$

**Returns:**
- Damped pseudoinverse

**Mathematical Operation:**
$$\mathbf{A}^+_\lambda = (\mathbf{A}^T \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^T$$

### Numerical Integration

#### `integrate_euler(f, y0: np.ndarray, t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]`

Euler method for ODE integration.

**Parameters:**
- `f`: Function $f(t, y)$ defining $\dot{y} = f(t, y)$
- `y0`: Initial condition
- `t_span`: Time span $(t_0, t_f)$
- `dt`: Time step

**Returns:**
- Tuple of (time_points, solution)

**Mathematical Operation:**
$$y_{n+1} = y_n + h f(t_n, y_n)$$

#### `integrate_runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]`

Fourth-order Runge-Kutta method.

**Parameters:**
- Same as `integrate_euler`

**Returns:**
- Tuple of (time_points, solution)

**Mathematical Operation:**
$$k_1 = h f(t_n, y_n)$$
$$k_2 = h f(t_n + h/2, y_n + k_1/2)$$
$$k_3 = h f(t_n + h/2, y_n + k_2/2)$$
$$k_4 = h f(t_n + h, y_n + k_3)$$
$$y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

## Linear Algebra

### Eigenvalue Problems

#### `solve_generalized_eigenvalue(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Solve generalized eigenvalue problem $\mathbf{A} \mathbf{v} = \lambda \mathbf{B} \mathbf{v}$.

**Parameters:**
- `A`, `B`: Input matrices

**Returns:**
- Tuple of (eigenvalues, eigenvectors)

#### `matrix_square_root(A: np.ndarray) -> np.ndarray`

Compute matrix square root using eigendecomposition.

**Parameters:**
- `A`: Symmetric positive definite matrix

**Returns:**
- Matrix square root $\mathbf{A}^{1/2}$

**Mathematical Operation:**
$$\mathbf{A}^{1/2} = \mathbf{Q} \mathbf{\Lambda}^{1/2} \mathbf{Q}^T$$

Where $\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$ is the eigendecomposition.

### Validation Functions

#### `is_rotation_matrix(R: np.ndarray, tolerance: float = 1e-10) -> bool`

Check if matrix is a valid rotation matrix.

**Parameters:**
- `R`: Matrix to check
- `tolerance`: Numerical tolerance

**Returns:**
- True if valid rotation matrix

**Validation Criteria:**
- Orthogonal: $\mathbf{R} \mathbf{R}^T = \mathbf{I}$
- Determinant: $\det(\mathbf{R}) = 1$
- Shape: $3 \times 3$

#### `is_symmetric(A: np.ndarray, tolerance: float = 1e-10) -> bool`

Check if matrix is symmetric.

**Parameters:**
- `A`: Matrix to check
- `tolerance`: Numerical tolerance

**Returns:**
- True if symmetric

**Validation Criterion:**
$$\|\mathbf{A} - \mathbf{A}^T\|_F < \text{tolerance}$$

#### `is_positive_definite(A: np.ndarray) -> bool`

Check if matrix is positive definite.

**Parameters:**
- `A`: Matrix to check

**Returns:**
- True if positive definite

**Validation Criterion:**
All eigenvalues $\lambda_i > 0$.

## Usage Examples

### Complete Transformation Chain

```python
import numpy as np
import py_pinocchio as pin

# Create rotation about Z-axis
angle = np.pi/4
axis = np.array([0, 0, 1])
R = pin.math.utils.rodrigues_formula(axis, angle)

# Create transformation
t = np.array([1.0, 2.0, 3.0])
T = pin.create_transform(R, t)

# Transform a point
point = np.array([1.0, 0.0, 0.0])
transformed_point = pin.math.transform.transform_point(T, point)

# Verify inverse
T_inv = pin.math.transform.invert_transform(T)
identity = pin.math.transform.compose_transforms(T, T_inv)

print(f"Original point: {point}")
print(f"Transformed point: {transformed_point}")
print(f"Identity check: {np.allclose(identity.rotation, np.eye(3))}")
```

### Spatial Algebra Example

```python
# Create spatial vectors
omega1 = np.array([1.0, 0.0, 0.0])
v1 = np.array([0.0, 1.0, 0.0])
spatial_v1 = pin.math.spatial.create_spatial_vector(omega1, v1)

omega2 = np.array([0.0, 1.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0])
spatial_v2 = pin.math.spatial.create_spatial_vector(omega2, v2)

# Compute spatial cross product
cross_product = pin.math.spatial.spatial_cross_product(spatial_v1, spatial_v2)

print(f"Spatial cross product: {cross_product}")
```

This mathematical utilities API provides the foundation for all geometric and numerical computations in py-pinocchio.
