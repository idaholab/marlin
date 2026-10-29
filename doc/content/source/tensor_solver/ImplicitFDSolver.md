# ImplicitFDSolver

!syntax description /TensorSolver/ImplicitFDSolver

## Overview

Solves each field $u$ in [!param](/TensorSolver/ImplicitFDSolver/buffer) implicitly from

!equation
\sum_i A_i u = \sum_i r_i + f,

where the $A_i$ are the finite-difference operators listed in
[!param](/TensorSolver/ImplicitFDSolver/operators) (as written, with no sign change), the
$r_i$ their right-hand side contributions, and $f$ the optional explicit
[!param](/TensorSolver/ImplicitFDSolver/source). Backward Euler diffusion
$\partial_t u = D \nabla^2 u + f$ is written with a [TimeDerivativeOperator.md] (action
$u/\Delta t$, contribution $u^n/\Delta t$) and a [FiniteDifferenceLaplacian.md] with `factor = -D`.

The operator is applied matrix-free with each operator's own kernel. The system must be symmetric
positive definite, or positive semidefinite with the constants as its nullspace (a steady periodic
Laplacian). In the singular case the right-hand side must have zero mean up to round-off, and the
zero-mean solution is returned.

The solve stops when the unpreconditioned residual satisfies
$\|b - A u\|_2 \le$ [!param](/TensorSolver/ImplicitFDSolver/l_tol) $\|b\|_2$, starting from the
previous solution. When a solve does not converge within
[!param](/TensorSolver/ImplicitFDSolver/l_max_its) iterations, a
`TensorSolveIterationAdaptiveDT` time stepper retries the time step with a smaller time step
(with one substep per time step); with any other time stepper it is an error. After the solve, the
operators' output buffers hold the operators applied to the new solution.

The [!param](/TensorSolver/ImplicitFDSolver/root_compute) is evaluated before every solve to
update the explicit terms, such as the source. The field operators in it are skipped, since the
solve applies them itself.

The `PETSC` linear solver (the default except on MPS) is PETSc's conjugate gradient method
preconditioned by geometric multigrid (`KSPCG` + `PCMG`), in double precision on CPU or CUDA. The
fine level applies the operators matrix-free; the coarse levels are the operators rediscretised on
grids coarsened by two per axis and assembled, connected by cell-centred multilinear interpolation.
Every level but the coarsest, which is solved by LU, is smoothed by two sweeps of the `smoother`.
The default, `CHEBYSHEV`, is PCMG's default Chebyshev smoother, preconditioned by SOR on the coarse
levels on CPU and by Jacobi on the matrix-free fine level and on every level on CUDA (PETSc's SOR
runs on the host). Chebyshev estimates each level's eigenvalue bounds with a few Krylov steps, again
on every level whenever the operators change (e.g. with the time step). `smoother = DAMPED_JACOBI`
needs no estimate: it is damped Jacobi
(`KSPRICHARDSON` + `PCJACOBI`) with the weight $\omega = 4/(3\rho)$, where
$\rho = 1 + \sum_{o \ne 0} |a_o| / a_0$ is the Gershgorin bound on the Jacobi scaled spectrum from
the level's stencil coefficients $a_o$, the maximum over the rows including those next to a
boundary face. The weight is recomputed when the operators change (e.g.
with the time step), unless `-<buffer>_mg_levels_ksp_richardson_scale` is set. It takes more
iterations on the biharmonic, and their number grows with the grid size.
The divergence test is off by default (`-<buffer>_ksp_divtol` sets one), as in `TORCH_PCG`: the
solve is warm started from the old solution and its residual compared to $\|b\|$, which scales as
$1/\Delta t$, so at large time steps the first residuals exceed PETSc's default divergence
tolerance on a converging solve.
PETSc options apply with the field name as prefix: `petsc_options_iname = '-pc_type'` with
`petsc_options_value = 'jacobi'` sets `-u_pc_type jacobi` for field `u`. The block options apply to
every field, and command line options (e.g. `-u_mg_levels_ksp_max_it 3`) override them.

The `TORCH_PCG` linear solver is a Jacobi preconditioned conjugate gradient method in torch. It runs
on every device, in single precision on devices without double precision support.

## Boundary conditions

On a non-periodic axis, each face of $u$ has a [TensorDirichletBC.md], [TensorNeumannBC.md] or
[TensorRobinBC.md] (see the [BCs system](syntax/TensorComputes/BCs/index.md) for face values and
their timing). With boundary values the operator action is affine, $A u + g$, and the solver solves
$A u = b - g$, with $g$ the action on $u = 0$, recomputed every solve. A Dirichlet face, or a Robin face with $a \ne 0$, makes the system
definite; with Neumann faces only it keeps the constants as its nullspace, and the right-hand
side, including the fluxes, must then have zero mean up to round-off. Source and fluxes that
balance in the continuous problem generally leave a discrete mean of order $h^2$, which is an
error: the data must balance on the grid. Supported are the time derivative and the Laplacian,
with Dirichlet, Neumann and Robin faces for the 3-point stencil and Dirichlet and Neumann faces
for the 5-point stencil, whose second ghost layer makes a Robin operator nonsymmetric. A Robin
face with $a b < 0$ is an error: its ghost term $a / (a/2 + b/\Delta x)$ turns negative once
$\Delta x < 2 |b / a|$, after which the operator may be indefinite, depending on the rest of the
problem, so whether it could be solved would depend on the grid. The biharmonic on a
non-periodic domain is an error too.

On the coarse multigrid levels, a stencil entry that reaches a ghost cell is folded onto the
mirrored interior cell, times the ghost factor $-(a/2 - b/s)/(a/2 + b/s)$ of the homogeneous
condition with $s = (2l+1) h$ for ghost layer $l$ and level spacing $h$ ($-1$ for Dirichlet, $1$
for Neumann). The interpolation folds the same way. This affects only the preconditioner.

## Example Input File Syntax

!listing test/tests/implicit_fd/be_mode.i block=TensorSolver

With boundary conditions whose values vary along the faces:

!listing test/tests/implicit_fd/bc_mms.i block=TensorComputes/BCs

!syntax parameters /TensorSolver/ImplicitFDSolver

!syntax inputs /TensorSolver/ImplicitFDSolver

!syntax children /TensorSolver/ImplicitFDSolver
