# LBMConvectiveOutflow

!syntax description /TensorComputes/Boundary/LBMConvectiveOutflow

Implements a convective outflow boundary condition for Lattice Boltzmann simulations.

## Overview

This boundary condition applies the discrete form of the convective transport equation at the outlet:

$$\frac{\partial f_i}{\partial t} + U_c \frac{\partial f_i}{\partial \hat{n}} = 0$$

which is discretized as:

$$f_i(\mathbf{x}_b, t) = \frac{f_i(\mathbf{x}_b, t-1) + U_c \, f_i(\mathbf{x}_n, t)}{1 + U_c}$$

where $\mathbf{x}_b$ is the boundary node, $\mathbf{x}_n$ is the first interior neighbor, and $U_c$
is the convection velocity.

The convective outflow BC allows vortices and other flow structures to leave the domain with
minimal reflections, making it suitable for simulations with unsteady wake dynamics
(e.g., vortex shedding behind bluff bodies).

### Convection Velocity

The convection velocity $U_c$ is controlled by
[!param](/TensorComputes/Boundary/LBMConvectiveOutflow/convection_velocity):

- `auto` (default): $U_c$ is computed on-the-fly as the mean absolute normal velocity at the
  first interior neighbor plane. This adapts automatically to the local flow conditions.
- A constant name or numeric value: $U_c$ is fixed to the user-specified value. This is useful
  when the mean outlet velocity is known a priori (e.g., from the inlet velocity in an
  incompressible channel flow).

### Old State

The boundary condition requires the old distribution function via
[!param](/TensorComputes/Boundary/LBMConvectiveOutflow/f_old), which provides $f_i(\mathbf{x}_b, t-1)$.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`).

## Example Input File Syntax

!listing test/tests/lbm/convective_outflow_2d_right.i block=TensorComputes/Boundary/right

!listing test/tests/lbm/convective_outflow_3d_bottom.i block=TensorComputes/Boundary/bottom

!syntax parameters /TensorComputes/Boundary/LBMConvectiveOutflow

!syntax inputs /TensorComputes/Boundary/LBMConvectiveOutflow

!syntax children /TensorComputes/Boundary/LBMConvectiveOutflow
