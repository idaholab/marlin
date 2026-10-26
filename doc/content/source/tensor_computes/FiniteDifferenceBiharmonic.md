# FiniteDifferenceBiharmonic

!syntax description /TensorComputes/Solve/FiniteDifferenceBiharmonic

## Overview

Computes `factor` times the biharmonic $\nabla^2\nabla^2 u$ of the input buffer
in real space. The stencil is the discrete Laplacian of
[FiniteDifferenceLaplacian.md] composed with itself
([!param](/TensorComputes/Solve/FiniteDifferenceBiharmonic/stencil_width) selects
the 3-point or 5-point Laplacian), which in 2D with the 3-point Laplacian is
the 13-point stencil. It reaches twice as far as the Laplacian, so two (3-point)
or four (5-point) ghost layers are exchanged before it is applied.

The operator also exposes its stencil for implicit solvers. On a periodic grid
the square of the symmetric Laplacian is positive semidefinite with the
constants as its nullspace, so with `factor > 0` the operator can be solved
implicitly with [ImplicitFDSolver.md], for example as the $M\kappa\nabla^4 c$
term of the Cahn-Hilliard equation.

On a non-periodic domain [ImplicitFDSolver.md] supports the 3-point
biharmonic with two conditions on each face, set by one boundary condition on
$u$:

- a [TensorNeumannBC.md] gives $\partial u/\partial n = q$ and
  $\partial(\nabla^2 u)/\partial n = 0$ (no flux of Cahn-Hilliard's chemical
  potential through the face, with $q = 0$);
- a [TensorDirichletBC.md] gives $u = g$ and $\nabla^2 u = 0$.

With such faces the operator is applied in two stages: the Laplacian of $u$
with its boundary condition gives $w = \nabla^2 u$, whose ghosts are then set to
the reflection ($+1$ for Neumann, $-1$ for Dirichlet) of its values next to the
face, and the Laplacian of $w$ is the result. So the second condition holds for
face data that vary along the face too. The operator is the square of the
Laplacian with the homogeneous condition and stays symmetric. A Robin face, and
the 5-point biharmonic on a non-periodic domain, are errors.

## Example Input File Syntax

!listing test/tests/implicit_fd/biharmonic_mode.i block=TensorComputes/Solve/bih

With no-flux faces (PFHub benchmark 1b):

!listing test/tests/implicit_fd/pfhub_1b.i block=TensorComputes/BCs

!syntax parameters /TensorComputes/Solve/FiniteDifferenceBiharmonic

!syntax inputs /TensorComputes/Solve/FiniteDifferenceBiharmonic

!syntax children /TensorComputes/Solve/FiniteDifferenceBiharmonic
