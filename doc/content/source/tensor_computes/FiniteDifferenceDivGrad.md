# FiniteDifferenceDivGrad

!syntax description /TensorComputes/Solve/FiniteDifferenceDivGrad

## Overview

Computes `factor` times $\nabla\cdot(D\nabla u)$ in real space for a cell-centred
coefficient buffer $D$
([!param](/TensorComputes/Solve/FiniteDifferenceDivGrad/coefficient)), as the
difference of the face fluxes on each axis:

!equation
\left[\nabla\cdot(D\nabla u)\right]_i = \sum_{d}
\frac{D_{i+\frac12}(u_{i+1} - u_i) - D_{i-\frac12}(u_i - u_{i-1})}{h_d^2}.

It serves diffusion with a variable diffusivity, Poisson problems with a variable
permittivity and variable mobility alike. With a constant $D$ it is
$D$ times the 3-point [FiniteDifferenceLaplacian.md].

### Face coefficient

[!param](/TensorComputes/Solve/FiniteDifferenceDivGrad/face_mean) selects $D$ on
the face between cells $i$ and $j$:

- `harmonic` (the default), $2D_iD_j/(D_i + D_j)$: the exact series conductance
  of two half cells, so a jump in $D$ on a cell face (e.g. two dielectric slabs)
  gives the exact flux. It requires $D_iD_j > 0$ on every face, which is checked;
  a face where $D$ reaches zero would be shut.
- `arithmetic`, $(D_i + D_j)/2$: for a coefficient that can reach zero, such as a
  degenerate mobility, where the harmonic mean would close the face.

### Sign of $D$

Applied explicitly the operator accepts $D$ of either sign with the arithmetic
mean. A negative coefficient is useful e.g. as the explicit part of an IMEX
split. In an implicit solve ([ImplicitFDSolver.md]) the operator must be
semidefinite: $D < 0$ is an error, and so is $D = 0$ with the harmonic mean. Both
are checked at every assembly.

### Domain faces

The ghost values of $u$ come from the boundary conditions of $u$
([TensorDirichletBC.md], [TensorNeumannBC.md], [TensorRobinBC.md]). The face
coefficient on a domain face is the boundary cell's own $D$; $D$ needs no boundary
condition. This is exact where $D$ is constant next to the face (e.g. at an
electrode). For a smoothly varying $D$ it is second order on Dirichlet and no-flux
faces, but first order on a Neumann face with a nonzero flux. A Neumann
condition gives $\partial u/\partial n$, so the flux through such a face is $D$ of
the boundary cell times the given value.

## Example Input File Syntax

!listing test/tests/real_space/divgrad.i block=TensorComputes/Solve/divgrad

!syntax parameters /TensorComputes/Solve/FiniteDifferenceDivGrad

!syntax inputs /TensorComputes/Solve/FiniteDifferenceDivGrad

!syntax children /TensorComputes/Solve/FiniteDifferenceDivGrad
