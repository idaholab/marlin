# LBMFixedFirstOrderBC

!syntax description /TensorComputes/Boundary/LBMFixedFirstOrderBC

LBMFixedFirstOrderBC implements Zou\-He velocity boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils.

## Overview

Enforces first\-order accurate macroscopic velocity at selected domain faces via Zou\-He formulas.
Choose faces with [!param](/TensorComputes/Boundary/LBMFixedFirstOrderBC/boundary) and provide
macroscopic fields as required by the implementation.

### Zou-He Boundary Condition

The Zou-He boundary condition is a popular method for specifying velocity or pressure boundaries in LBM. It works by using the "bounce-back" of the non-equilibrium part of the distribution function to determine the unknown populations at the boundary.

For a velocity boundary (e.g., specified $u_x$ at a left inlet), the unknown distributions pointing into the domain are calculated to satisfy the macroscopic constraints (density and velocity).

For example, for a D2Q9 lattice at a left boundary ($x=0$) with specified velocity $\mathbf{u} = (u_x, 0)$, the unknown populations $f_1, f_5, f_8$ (pointing right) are determined from the known populations $f_3, f_6, f_7$ (pointing left) and the known $f_0, f_2, f_4$ (vertical and rest).

The density $\rho$ at the boundary is first computed as:

$$
\rho = \frac{1}{1 - u_x} \left( f_0 + f_2 + f_4 + 2(f_3 + f_6 + f_7) \right)
$$

Then, the unknown populations are set:

$$
f_1 = f_3 + \frac{2}{3} \rho u_x
$$

$$
f_5 = f_7 - \frac{1}{2} (f_2 - f_4) + \frac{1}{6} \rho u_x
$$

$$
f_8 = f_6 + \frac{1}{2} (f_2 - f_4) + \frac{1}{6} \rho u_x
$$

Similar equations are derived for other boundaries (right, top, bottom) and for 3D stencils, ensuring mass and momentum conservation at the boundary.

> **Note**: Specializations for D2Q9 are implemented for better performance and accuracy. General implementations for other stencils (like D3Q19, D3Q27) follow the same principles but with generalized formulations.

## Example Input File Syntax

!listing
[TensorComputes]
  [Boundary]
    [left]
      type = LBMFixedFirstOrderBC
      buffer = f
      f = f
      value = 0.0001
      boundary = left
    []
    [right]
      type = LBMFixedFirstOrderBC
      buffer = f
      f = f
      value = 0.00011
      boundary = right
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Boundary/LBMFixedFirstOrderBC
