# LBMFixedZerothOrderBC

!syntax description /TensorComputes/Boundary/LBMFixedZerothOrderBC

LBMFixedZerothOrderBC implements Zou-He pressure boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils.

## Overview

Enforces first\-order accurate macroscopic pressure at selected domain faces via Zou\-He formulas.
Choose faces with [!param](/TensorComputes/Boundary/LBMFixedZerothOrderBC/boundary) and provide macroscopic fields as
required by the implementation.

### Zou-He Pressure Boundary Condition

Similar to the velocity boundary condition, the Zou-He pressure (or density) boundary condition uses the bounce-back rule for the non-equilibrium part of the distribution to determine unknown populations.

For a pressure boundary (e.g., specified density $\rho_{in}$ at a left inlet), the velocity $u_x$ is unknown and must be determined from the known populations and the specified density.

For a D2Q9 lattice at a left boundary ($x=0$) with specified density $\rho_{in}$, the velocity $u_x$ is calculated as:

$$
u_x = 1 - \frac{f_0 + f_2 + f_4 + 2(f_3 + f_6 + f_7)}{\rho_{in}}
$$

Once $u_x$ is known, the unknown populations $f_1, f_5, f_8$ (pointing right) are determined using the same relations as in the velocity boundary condition:

$$
f_1 = f_3 + \frac{2}{3} \rho_{in} u_x
$$

$$
f_5 = f_7 - \frac{1}{2} (f_2 - f_4) + \frac{1}{6} \rho_{in} u_x
$$

$$
f_8 = f_6 + \frac{1}{2} (f_2 - f_4) + \frac{1}{6} \rho_{in} u_x
$$

This ensures that the macroscopic density at the boundary matches the specified value $\rho_{in}$ while the velocity adjusts to satisfy mass conservation.

> **Note**: Specializations for D2Q9 are implemented for better performance. General implementations for other stencils follow similar principles.

## Example Input File Syntax

!listing
[TensorComputes]
  [Boundary]
    [left]
      type = LBMFixedZerothOrderBC
      buffer = f
      f = f
      value = 1.1
      boundary = left
    []
    [right]
      type = LBMFixedZerothOrderBC
      buffer = f
      f = f
      value = 1.00000
      boundary = right
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Boundary/LBMFixedZerothOrderBC
