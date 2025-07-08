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

4×4 matrices representing rigid body transformations:

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
```
q_{k+1} = q_k + J†(q_k) Δx
```

Where:
- `J†` is pseudoinverse of Jacobian
- `Δx = x_desired - x_current`

**Damped least squares**:
```
q_{k+1} = q_k + (JᵀJ + λI)⁻¹Jᵀ Δx
```

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

**Coriolis matrix** `C(q,q̇)`:
```
Cᵢⱼ = Σₖ Γᵢⱼₖ q̇ₖ
```

Where Christoffel symbols:
```
Γᵢⱼₖ = ½ (∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ - ∂Mⱼₖ/∂qᵢ)
```

**Properties**:
- `Ṁ - 2C` is skew-symmetric
- Quadratic in joint velocities
- Vanishes when `q̇ = 0`

### Gravity Forces

Gravity vector in joint space:
```
g(q) = -Σᵢ JᵢᵀMᵢ g₀
```

Where `g₀ = [0; 0; 0; 0; 0; -9.81]` is spatial gravity vector.

**Properties**:
- Independent of joint velocities
- Configuration dependent
- Conservative force (gradient of potential energy)

## Computational Algorithms

### Recursive Newton-Euler Algorithm (RNEA)

Efficient O(n) algorithm for inverse dynamics:

**Forward pass** (kinematics):
```
For i = 1 to n:
    vᵢ = Xᵢ vᵢ₋₁ + Sᵢ q̇ᵢ
    aᵢ = Xᵢ aᵢ₋₁ + Sᵢ q̈ᵢ + vᵢ ×* Sᵢ q̇ᵢ
```

**Backward pass** (dynamics):
```
For i = n to 1:
    fᵢ = Iᵢ aᵢ + vᵢ ×* (Iᵢ vᵢ) + Xᵢ₊₁ᵀ fᵢ₊₁
    τᵢ = Sᵢᵀ fᵢ
```

Where:
- `Xᵢ` is spatial transformation from link i-1 to i
- `Sᵢ` is motion subspace for joint i
- `vᵢ, aᵢ` are spatial velocity and acceleration
- `fᵢ` is spatial force
- `Iᵢ` is spatial inertia

### Articulated Body Algorithm (ABA)

Efficient O(n) algorithm for forward dynamics:

**Forward pass**:
```
For i = 1 to n:
    vᵢ = Xᵢ vᵢ₋₁ + Sᵢ q̇ᵢ
    cᵢ = vᵢ ×* Sᵢ q̇ᵢ
    Iᵢᴬ = Iᵢ
```

**Backward pass**:
```
For i = n to 1:
    Uᵢ = Iᵢᴬ Sᵢ
    Dᵢ = Sᵢᵀ Uᵢ
    uᵢ = τᵢ - Sᵢᵀ pᵢᴬ
    
    If i > 1:
        Iᵢ₋₁ᴬ += Xᵢᵀ (Iᵢᴬ - UᵢDᵢ⁻¹Uᵢᵀ) Xᵢ
        pᵢ₋₁ᴬ += Xᵢᵀ (pᵢᴬ + Iᵢᴬ cᵢ + UᵢDᵢ⁻¹uᵢ)
```

**Forward pass** (accelerations):
```
For i = 1 to n:
    q̈ᵢ = (uᵢ - Uᵢᵀ aᵢ₋₁) / Dᵢ
    aᵢ = Xᵢ aᵢ₋₁ + Sᵢ q̈ᵢ + cᵢ
```

### Composite Rigid Body Algorithm (CRBA)

Efficient O(n²) algorithm for mass matrix:

```
For i = 1 to n:
    Iᵢᶜ = Iᵢ
    
For i = n to 1:
    If i > 1:
        Iᵢ₋₁ᶜ += Xᵢᵀ Iᵢᶜ Xᵢ
    
    fᵢ = Iᵢᶜ Sᵢ
    Mᵢᵢ = Sᵢᵀ fᵢ
    
    j = i
    While j > 1:
        fⱼ₋₁ = Xⱼᵀ fⱼ
        Mᵢⱼ₋₁ = Sⱼ₋₁ᵀ fⱼ₋₁
        j = j - 1
```

### Jacobian Computation

**Geometric Jacobian**:
```
For i = 1 to n:
    If joint i is revolute:
        Jᵢ = [zᵢ₋₁; zᵢ₋₁ × (pₙ - pᵢ₋₁)]
    Else if joint i is prismatic:
        Jᵢ = [0; zᵢ₋₁]
```

**Time derivative**:
```
J̇ = Σᵢ (∂J/∂qᵢ) q̇ᵢ
```

## Numerical Considerations

### Stability

- Use double precision for better accuracy
- Avoid matrix inversions when possible
- Check condition numbers of matrices
- Handle singular configurations gracefully

### Efficiency

- Exploit sparsity in robot structure
- Use recursive algorithms (O(n) vs O(n³))
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
