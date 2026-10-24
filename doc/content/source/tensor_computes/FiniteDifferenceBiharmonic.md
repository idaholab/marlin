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

## Example Input File Syntax

!listing test/tests/implicit_fd/biharmonic_mode.i block=TensorComputes/Solve/bih

!syntax parameters /TensorComputes/Solve/FiniteDifferenceBiharmonic

!syntax inputs /TensorComputes/Solve/FiniteDifferenceBiharmonic

!syntax children /TensorComputes/Solve/FiniteDifferenceBiharmonic
