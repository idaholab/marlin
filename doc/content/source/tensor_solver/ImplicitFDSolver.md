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
the level's stencil coefficients $a_o$. The weight is recomputed when the operators change (e.g.
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

## Example Input File Syntax

!listing test/tests/implicit_fd/be_mode.i block=TensorSolver

!syntax parameters /TensorSolver/ImplicitFDSolver

!syntax inputs /TensorSolver/ImplicitFDSolver

!syntax children /TensorSolver/ImplicitFDSolver
