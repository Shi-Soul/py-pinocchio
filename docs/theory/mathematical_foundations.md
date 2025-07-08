# Mathematical Foundations of Robot Dynamics

This document provides the mathematical theory underlying py-pinocchio's algorithms. Understanding these concepts will help you use the library effectively and debug issues.

## Table of Contents

1. [Spatial Algebra](#spatial-algebra)
2. [Rigid Body Kinematics](#rigid-body-kinematics)
3. [Robot Kinematics](#robot-kinematics)
4. [Rigid Body Dynamics](#rigid-body-dynamics)
5. [Robot Dynamics](#robot-dynamics)
6. [Computational Algorithms](#computational-algorithms)

## Spatial Algebra

### Spatial Vectors

Spatial vectors combine linear and angular quantities in 6D space:

$$\mathbf{v} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} \in \mathbb{R}^6$$

Where:
- $\boldsymbol{\omega} \in \mathbb{R}^3$ is angular velocity/acceleration
- $\mathbf{v} \in \mathbb{R}^3$ is linear velocity/acceleration

**Types of spatial vectors:**
- **Motion vectors**: velocities, accelerations
- **Force vectors**: forces and torques

### Spatial Transformations

A spatial transformation matrix relates spatial vectors between frames:

$$\mathbf{X} = \begin{bmatrix} \mathbf{R} & \mathbf{0} \\ \mathbf{r}^\times \mathbf{R} & \mathbf{R} \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

Where:
- $\mathbf{R} \in SO(3)$ is rotation matrix
- $\mathbf{r} \in \mathbb{R}^3$ is translation vector
- $\mathbf{r}^\times$ is the skew-symmetric matrix of $\mathbf{r}$

**Properties:**
- $\mathbf{X}^{-1} = \begin{bmatrix} \mathbf{R}^T & \mathbf{0} \\ -\mathbf{R}^T \mathbf{r}^\times & \mathbf{R}^T \end{bmatrix}$
- Motion transformation: $\mathbf{v}_2 = \mathbf{X}_{21} \mathbf{v}_1$
- Force transformation: $\mathbf{f}_1 = \mathbf{X}_{21}^T \mathbf{f}_2$

### Spatial Cross Product

The spatial cross product captures Coriolis effects:

$$\mathbf{v}_1 \times^* \mathbf{v}_2 = \begin{bmatrix} \boldsymbol{\omega}_1 \times \boldsymbol{\omega}_2 + \mathbf{v}_1 \times \mathbf{v}_2 \\ \boldsymbol{\omega}_1 \times \mathbf{v}_2 \end{bmatrix}$$

**Properties:**
- Bilinear: $(a\mathbf{v}_1 + b\mathbf{v}_2) \times^* \mathbf{v}_3 = a(\mathbf{v}_1 \times^* \mathbf{v}_3) + b(\mathbf{v}_2 \times^* \mathbf{v}_3)$
- Anti-symmetric: $\mathbf{v}_1 \times^* \mathbf{v}_2 = -(\mathbf{v}_2 \times^* \mathbf{v}_1)$

### Spatial Inertia

Spatial inertia combines mass and rotational inertia:

$$\mathbf{I} = \begin{bmatrix} \mathbf{I}_c + m \mathbf{c}^\times \mathbf{c}^\times & m \mathbf{c}^\times \\ m \mathbf{c}^\times & m \mathbf{1} \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

Where:
- $\mathbf{I}_c \in \mathbb{R}^{3 \times 3}$ is inertia tensor about center of mass
- $m \in \mathbb{R}$ is mass
- $\mathbf{c} \in \mathbb{R}^3$ is center of mass position
- $\mathbf{1} \in \mathbb{R}^{3 \times 3}$ is identity matrix

**Properties:**
- Symmetric: $\mathbf{I} = \mathbf{I}^T$
- Positive definite for physical bodies
- Transforms as: $\mathbf{I}_2 = \mathbf{X}_{21}^T \mathbf{I}_1 \mathbf{X}_{21}$

## Rigid Body Kinematics

### Rotation Representations

**Rotation Matrix** $\mathbf{R} \in SO(3)$:
- Orthogonal: $\mathbf{R}\mathbf{R}^T = \mathbf{I}$
- Determinant: $\det(\mathbf{R}) = 1$
- 9 parameters, 3 DOF

**Axis-Angle** $(\mathbf{k}, \theta)$:
- Unit axis: $\mathbf{k} \in S^2$
- Angle: $\theta \in \mathbb{R}$
- Rodrigues formula: $\mathbf{R} = \mathbf{I} + \sin(\theta)[\mathbf{k}]_\times + (1-\cos(\theta))[\mathbf{k}]_\times^2$

**Quaternions** $\mathbf{q} = (w, x, y, z)$:
- Unit quaternion: $|\mathbf{q}| = 1$
- Rotation matrix: $\mathbf{R}(\mathbf{q}) = \mathbf{I} + 2w[\mathbf{v}]_\times + 2[\mathbf{v}]_\times^2$
- Where $\mathbf{v} = (x, y, z)$

### Homogeneous Transformations

$4 \times 4$ matrices representing rigid body transformations:

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)$$

**Properties:**
- Composition: $\mathbf{T}_3 = \mathbf{T}_1 \mathbf{T}_2$
- Inverse: $\mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T\mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$
- Point transformation: $\mathbf{p}' = \mathbf{T}\mathbf{p} = \mathbf{R}\mathbf{p} + \mathbf{t}$

### Velocity and Acceleration

**Angular velocity** relates rotation rates:
$$\dot{\mathbf{R}} = [\boldsymbol{\omega}]_\times \mathbf{R}$$

**Spatial velocity** in body frame:
$$\mathbf{v} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} \text{ where } \begin{cases} \boldsymbol{\omega} = \text{angular velocity} \\ \mathbf{v} = \text{linear velocity} \end{cases}$$

**Spatial acceleration**:
$$\mathbf{a} = \begin{bmatrix} \boldsymbol{\alpha} \\ \mathbf{a} \end{bmatrix} \text{ where } \begin{cases} \boldsymbol{\alpha} = \text{angular acceleration} \\ \mathbf{a} = \text{linear acceleration} \end{cases}$$

## Robot Kinematics

### Forward Kinematics

Maps joint angles to end-effector pose:

$$\mathbf{T}_0^n(\mathbf{q}) = \mathbf{T}_0^1(q_1) \mathbf{T}_1^2(q_2) \cdots \mathbf{T}_{n-1}^n(q_n)$$

Where $\mathbf{T}_i^{i+1}(q_{i+1})$ is the transformation from link $i$ to link $i+1$.

**For revolute joint**:
$$\mathbf{T}_i^{i+1}(q_{i+1}) = \begin{bmatrix} \mathbf{R}_z(q_{i+1}) & \mathbf{p}_{i+1} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**For prismatic joint**:
$$\mathbf{T}_i^{i+1}(q_{i+1}) = \begin{bmatrix} \mathbf{I} & \mathbf{p}_i + q_{i+1} \mathbf{z}_i \\ \mathbf{0}^T & 1 \end{bmatrix}$$

### Jacobian Matrices

The Jacobian relates joint velocities to end-effector velocity:

$$\mathbf{v} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

**Geometric Jacobian** (body frame):
$$\mathbf{J} = \begin{bmatrix} \mathbf{J}_1 & \mathbf{J}_2 & \cdots & \mathbf{J}_n \end{bmatrix}$$

For revolute joint $i$:
$$\mathbf{J}_i = \begin{bmatrix} \mathbf{z}_{i-1} \\ \mathbf{z}_{i-1} \times (\mathbf{p}_n - \mathbf{p}_{i-1}) \end{bmatrix}$$

For prismatic joint $i$:
$$\mathbf{J}_i = \begin{bmatrix} \mathbf{0} \\ \mathbf{z}_{i-1} \end{bmatrix}$$

**Analytical Jacobian** relates to Euler angle rates:
$$\begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} = \begin{bmatrix} \mathbf{E}(\boldsymbol{\phi}) & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{bmatrix} \begin{bmatrix} \dot{\boldsymbol{\phi}} \\ \dot{\mathbf{p}} \end{bmatrix}$$

### Inverse Kinematics

Find joint angles for desired end-effector pose:

**Newton-Raphson method**:
$$\mathbf{q}_{k+1} = \mathbf{q}_k + \mathbf{J}^{\dagger}(\mathbf{q}_k) \Delta \mathbf{x}$$

Where:
- $\mathbf{J}^{\dagger}$ is the Moore-Penrose pseudoinverse of the Jacobian
- $\Delta \mathbf{x} = \mathbf{x}_{\text{desired}} - \mathbf{x}_{\text{current}}$ is the pose error

**Damped least squares**:
$$\mathbf{q}_{k+1} = \mathbf{q}_k + (\mathbf{J}^T\mathbf{J} + \lambda \mathbf{I})^{-1}\mathbf{J}^T \Delta \mathbf{x}$$

Where $\lambda > 0$ is the damping parameter for numerical stability near singularities.

## Rigid Body Dynamics

### Newton-Euler Equations

For a single rigid body:

$$\mathbf{F} = m\mathbf{a}_c \quad \text{(linear motion)}$$
$$\boldsymbol{\tau} = \mathbf{I}_c \boldsymbol{\alpha} + \boldsymbol{\omega} \times (\mathbf{I}_c \boldsymbol{\omega}) \quad \text{(angular motion)}$$

Where:
- $\mathbf{F}$ is resultant force
- $\boldsymbol{\tau}$ is resultant torque about center of mass
- $\mathbf{a}_c$ is linear acceleration of center of mass
- $\boldsymbol{\alpha}$ is angular acceleration
- $\mathbf{I}_c$ is inertia tensor about center of mass

### Spatial Form

In spatial notation:
$$\mathbf{f} = \mathbf{I} \mathbf{a} + \mathbf{v} \times^* (\mathbf{I} \mathbf{v})$$

Where:
- $\mathbf{f}$ is spatial force
- $\mathbf{I}$ is spatial inertia
- $\mathbf{a}$ is spatial acceleration
- $\mathbf{v}$ is spatial velocity

### Energy and Momentum

**Kinetic energy**:
$$T = \frac{1}{2} \mathbf{v}^T \mathbf{I} \mathbf{v}$$

**Linear momentum**:
$$\mathbf{p} = m \mathbf{v}_c$$

**Angular momentum**:
$$\mathbf{L} = \mathbf{I}_c \boldsymbol{\omega} + m \mathbf{r}_c \times \mathbf{v}_c$$

## Robot Dynamics

### Equations of Motion

The robot equations of motion in joint space:

$$\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

Where:
- $\mathbf{M}(\mathbf{q})$ is the mass matrix
- $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$ is the Coriolis/centrifugal matrix
- $\mathbf{g}(\mathbf{q})$ is the gravity vector
- $\boldsymbol{\tau}$ is the joint torque vector

### Mass Matrix

The joint-space mass matrix:

$$\mathbf{M}(\mathbf{q}) = \sum_{i=1}^n \mathbf{J}_i^T \mathbf{M}_i \mathbf{J}_i$$

Where:
- $\mathbf{J}_i$ is the Jacobian for link $i$
- $\mathbf{M}_i$ is the spatial inertia of link $i$

**Properties**:
- Symmetric: $\mathbf{M} = \mathbf{M}^T$
- Positive definite
- Configuration dependent
- Size: $n \times n$ where $n$ = number of DOF

### Coriolis and Centrifugal Forces

**Coriolis matrix** $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$:
$$C_{ij} = \sum_{k=1}^n \Gamma_{ijk} \dot{q}_k$$

Where the Christoffel symbols are:
$$\Gamma_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$

**Properties**:
- $\dot{\mathbf{M}} - 2\mathbf{C}$ is skew-symmetric
- Quadratic in joint velocities
- Vanishes when $\dot{\mathbf{q}} = \mathbf{0}$

### Gravity Forces

Gravity vector in joint space:
$$\mathbf{g}(\mathbf{q}) = -\sum_{i=1}^n \mathbf{J}_i^T \mathbf{M}_i \mathbf{g}_0$$

Where $\mathbf{g}_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ -9.81 \end{bmatrix}$ is the spatial gravity vector in m/s².

**Properties**:
- Independent of joint velocities
- Configuration dependent
- Conservative force (gradient of potential energy)

## Computational Algorithms

### Recursive Newton-Euler Algorithm (RNEA)

Efficient $O(n)$ algorithm for inverse dynamics:

**Forward pass** (kinematics):
$$\begin{align}
\text{For } i &= 1 \text{ to } n: \\
\mathbf{v}_i &= \mathbf{X}_i \mathbf{v}_{i-1} + \mathbf{S}_i \dot{q}_i \\
\mathbf{a}_i &= \mathbf{X}_i \mathbf{a}_{i-1} + \mathbf{S}_i \ddot{q}_i + \mathbf{v}_i \times^* \mathbf{S}_i \dot{q}_i
\end{align}$$

**Backward pass** (dynamics):
$$\begin{align}
\text{For } i &= n \text{ to } 1: \\
\mathbf{f}_i &= \mathbf{I}_i \mathbf{a}_i + \mathbf{v}_i \times^* (\mathbf{I}_i \mathbf{v}_i) + \mathbf{X}_{i+1}^T \mathbf{f}_{i+1} \\
\tau_i &= \mathbf{S}_i^T \mathbf{f}_i
\end{align}$$

Where:
- $\mathbf{X}_i$ is spatial transformation from link $i-1$ to $i$
- $\mathbf{S}_i$ is motion subspace for joint $i$
- $\mathbf{v}_i, \mathbf{a}_i$ are spatial velocity and acceleration
- $\mathbf{f}_i$ is spatial force
- $\mathbf{I}_i$ is spatial inertia

### Articulated Body Algorithm (ABA)

Efficient $O(n)$ algorithm for forward dynamics:

**Forward pass**:
$$\begin{align}
\text{For } i &= 1 \text{ to } n: \\
\mathbf{v}_i &= \mathbf{X}_i \mathbf{v}_{i-1} + \mathbf{S}_i \dot{q}_i \\
\mathbf{c}_i &= \mathbf{v}_i \times^* \mathbf{S}_i \dot{q}_i \\
\mathbf{I}_i^A &= \mathbf{I}_i
\end{align}$$

**Backward pass**:
$$\begin{align}
\text{For } i &= n \text{ to } 1: \\
\mathbf{U}_i &= \mathbf{I}_i^A \mathbf{S}_i \\
D_i &= \mathbf{S}_i^T \mathbf{U}_i \\
u_i &= \tau_i - \mathbf{S}_i^T \mathbf{p}_i^A \\
\\
\text{If } i &> 1: \\
\mathbf{I}_{i-1}^A &\mathrel{+}= \mathbf{X}_i^T (\mathbf{I}_i^A - \mathbf{U}_i D_i^{-1} \mathbf{U}_i^T) \mathbf{X}_i \\
\mathbf{p}_{i-1}^A &\mathrel{+}= \mathbf{X}_i^T (\mathbf{p}_i^A + \mathbf{I}_i^A \mathbf{c}_i + \mathbf{U}_i D_i^{-1} u_i)
\end{align}$$

**Forward pass** (accelerations):
$$\begin{align}
\text{For } i &= 1 \text{ to } n: \\
\ddot{q}_i &= (u_i - \mathbf{U}_i^T \mathbf{a}_{i-1}) / D_i \\
\mathbf{a}_i &= \mathbf{X}_i \mathbf{a}_{i-1} + \mathbf{S}_i \ddot{q}_i + \mathbf{c}_i
\end{align}$$

### Composite Rigid Body Algorithm (CRBA)

Efficient $O(n^2)$ algorithm for mass matrix:

$$\begin{align}
\text{For } i &= 1 \text{ to } n: \\
\mathbf{I}_i^c &= \mathbf{I}_i \\
\\
\text{For } i &= n \text{ to } 1: \\
\text{If } i &> 1: \\
\mathbf{I}_{i-1}^c &\mathrel{+}= \mathbf{X}_i^T \mathbf{I}_i^c \mathbf{X}_i \\
\\
\mathbf{f}_i &= \mathbf{I}_i^c \mathbf{S}_i \\
M_{ii} &= \mathbf{S}_i^T \mathbf{f}_i \\
\\
j &= i \\
\text{While } j &> 1: \\
\mathbf{f}_{j-1} &= \mathbf{X}_j^T \mathbf{f}_j \\
M_{i,j-1} &= \mathbf{S}_{j-1}^T \mathbf{f}_{j-1} \\
j &= j - 1
\end{align}$$

### Jacobian Computation

**Geometric Jacobian**:
$$\begin{align}
\text{For } i &= 1 \text{ to } n: \\
\text{If joint } i \text{ is revolute:} \\
\mathbf{J}_i &= \begin{bmatrix} \mathbf{z}_{i-1} \\ \mathbf{z}_{i-1} \times (\mathbf{p}_n - \mathbf{p}_{i-1}) \end{bmatrix} \\
\text{Else if joint } i \text{ is prismatic:} \\
\mathbf{J}_i &= \begin{bmatrix} \mathbf{0} \\ \mathbf{z}_{i-1} \end{bmatrix}
\end{align}$$

**Time derivative**:
$$\dot{\mathbf{J}} = \sum_{i=1}^n \frac{\partial \mathbf{J}}{\partial q_i} \dot{q}_i$$

## Numerical Considerations

### Stability

- Use double precision for better accuracy
- Avoid matrix inversions when possible
- Check condition numbers of matrices
- Handle singular configurations gracefully

### Efficiency

- Exploit sparsity in robot structure
- Use recursive algorithms ($O(n)$ vs $O(n^3)$)
- Cache computed quantities when possible
- Vectorize operations for multiple configurations

### Validation

- Check conservation of energy
- Verify symmetry of mass matrix
- Test with known analytical solutions
- Compare with other implementations

## References

1. Featherstone, R. "Rigid Body Dynamics Algorithms" (2008)
2. Murray, R.M., Li, Z., Sastry, S.S. "A Mathematical Introduction to Robotic Manipulation" (1994)
3. Siciliano, B., Khatib, O. "Springer Handbook of Robotics" (2016)
4. Lynch, K.M., Park, F.C. "Modern Robotics" (2017)

This mathematical foundation provides the theoretical basis for understanding and extending py-pinocchio's capabilities.
