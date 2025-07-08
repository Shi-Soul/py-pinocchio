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

```
v₁ ×* v₂ = [ω₁ × ω₂ + v₁ × v₂]
           [ω₁ × v₂        ]
```

**Properties:**
- Bilinear: `(av₁ + bv₂) ×* v₃ = a(v₁ ×* v₃) + b(v₂ ×* v₃)`
- Anti-symmetric: `v₁ ×* v₂ = -(v₂ ×* v₁)`

### Spatial Inertia

Spatial inertia combines mass and rotational inertia:

```
I = [I_c + m c×c×  m c×]  ∈ ℝ⁶ˣ⁶
    [m c×          m 1 ]
```

Where:
- `I_c ∈ ℝ³ˣ³` is inertia tensor about center of mass
- `m ∈ ℝ` is mass
- `c ∈ ℝ³` is center of mass position
- `1 ∈ ℝ³ˣ³` is identity matrix

**Properties:**
- Symmetric: `I = Iᵀ`
- Positive definite for physical bodies
- Transforms as: `I₂ = X₂₁ᵀ I₁ X₂₁`

## Rigid Body Kinematics

### Rotation Representations

**Rotation Matrix** `R ∈ SO(3)`:
- Orthogonal: `RRᵀ = I`
- Determinant: `det(R) = 1`
- 9 parameters, 3 DOF

**Axis-Angle** `(k, θ)`:
- Unit axis: `k ∈ S²`
- Angle: `θ ∈ ℝ`
- Rodrigues formula: `R = I + sin(θ)[k]× + (1-cos(θ))[k]×²`

**Quaternions** `q = (w, x, y, z)`:
- Unit quaternion: `|q| = 1`
- Rotation matrix: `R(q) = I + 2w[v]× + 2[v]×²`
- Where `v = (x, y, z)`

### Homogeneous Transformations

4×4 matrices representing rigid body transformations:

```
T = [R t]  ∈ SE(3)
    [0 1]
```

**Properties:**
- Composition: `T₃ = T₁ T₂`
- Inverse: `T⁻¹ = [Rᵀ -Rᵀt]`
                  `[0   1  ]`
- Point transformation: `p' = Tp = Rp + t`

### Velocity and Acceleration

**Angular velocity** relates rotation rates:
```
Ṙ = [ω]× R
```

**Spatial velocity** in body frame:
```
v = [ω]  where ω = angular velocity
    [v]        v = linear velocity
```

**Spatial acceleration**:
```
a = [α]  where α = angular acceleration
    [a]        a = linear acceleration
```

## Robot Kinematics

### Forward Kinematics

Maps joint angles to end-effector pose:

```
T₀ⁿ(q) = T₀¹(q₁) T₁²(q₂) ... Tⁿ⁻¹ⁿ(qₙ)
```

Where `Tⁱⁱ⁺¹(qᵢ₊₁)` is the transformation from link i to link i+1.

**For revolute joint**:
```
Tⁱⁱ⁺¹(qᵢ₊₁) = [Rz(qᵢ₊₁) pᵢ₊₁]
               [0        1   ]
```

**For prismatic joint**:
```
Tⁱⁱ⁺¹(qᵢ₊₁) = [I  pᵢ + qᵢ₊₁ zᵢ]
               [0  1         ]
```

### Jacobian Matrices

The Jacobian relates joint velocities to end-effector velocity:

```
v = J(q) q̇
```

**Geometric Jacobian** (body frame):
```
J = [J₁ J₂ ... Jₙ]
```

For revolute joint i:
```
Jᵢ = [zᵢ₋₁        ]  (if joint affects rotation)
     [zᵢ₋₁ × (pₙ-pᵢ₋₁)]
```

For prismatic joint i:
```
Jᵢ = [0  ]
     [zᵢ₋₁]
```

**Analytical Jacobian** relates to Euler angle rates:
```
[ω] = [E(φ)  0 ] [φ̇]
[v]   [0    I ] [ṗ]
```

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

```
F = ma_c     (linear motion)
τ = I_c α + ω × (I_c ω)  (angular motion)
```

Where:
- `F` is resultant force
- `τ` is resultant torque about center of mass
- `a_c` is linear acceleration of center of mass
- `α` is angular acceleration
- `I_c` is inertia tensor about center of mass

### Spatial Form

In spatial notation:
```
f = I a + v ×* (I v)
```

Where:
- `f` is spatial force
- `I` is spatial inertia
- `a` is spatial acceleration
- `v` is spatial velocity

### Energy and Momentum

**Kinetic energy**:
```
T = ½ vᵀ I v
```

**Linear momentum**:
```
p = m v_c
```

**Angular momentum**:
```
L = I_c ω + m r_c × v_c
```

## Robot Dynamics

### Equations of Motion

The robot equations of motion in joint space:

```
M(q) q̈ + C(q,q̇) q̇ + g(q) = τ
```

Where:
- `M(q)` is the mass matrix
- `C(q,q̇)` is the Coriolis/centrifugal matrix
- `g(q)` is the gravity vector
- `τ` is the joint torque vector

### Mass Matrix

The joint-space mass matrix:

```
M(q) = Σᵢ JᵢᵀMᵢJᵢ
```

Where:
- `Jᵢ` is the Jacobian for link i
- `Mᵢ` is the spatial inertia of link i

**Properties**:
- Symmetric: `M = Mᵀ`
- Positive definite
- Configuration dependent
- Size: n×n where n = number of DOF

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
