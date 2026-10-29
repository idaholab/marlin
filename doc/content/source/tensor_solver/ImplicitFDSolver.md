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
$u/\Delta t$, contribution $u^n/\Delta t$) and a [FiniteDifferenceLaplacian.md] with `factor = -D`;
the time derivative's `order = 2` gives variable-step BDF2 instead.

The operator is applied matrix-free with each operator's own kernel. The system must be symmetric
positive definite, or positive semidefinite with the constants as its nullspace (a steady periodic
Laplacian). In the singular case the right-hand side must have zero mean up to round-off, and the
zero-mean solution is returned. The round-off is measured against the largest
$\sum |b|$ of the field's solves so far: a conserved source that decays (e.g. the charge of a
concentration whose fluctuations die out) keeps the round-off mean of its largest size, which
would otherwise exceed the shrinking current scale.

When the operator maps the constants to $\lambda > 0$ times the constants (the time derivative
with periodic or Neumann faces, e.g. diffusion or Cahn-Hilliard on a periodic or no-flux box),
summing the equations fixes the mean of the discrete solution, $\bar u = \bar b / \lambda$. The
solver detects this from the action of the operator on the constants and sets the mean of the
Krylov solution to $\bar b / \lambda$ after each solve, removing the solver error in that mode.
In double precision the total of a conserved field is then kept to round-off at any
[!param](/TensorSolver/ImplicitFDSolver/l_tol), rather than drifting by the solver tolerance every
step (a preconditioner mixes the constants with the other modes unless its diagonal is uniform,
as next to a boundary face or with SOR it is not). Nothing has to be marked:
the mean of a non-conserved field follows from the summed equations in the same way.

The fields are solved one after another in the order of
[!param](/TensorSolver/ImplicitFDSolver/buffer), with every explicit term, such as a source
computed from another field, evaluated once before the first solve (without
[!param](/TensorSolver/ImplicitFDSolver/nonlinear_solve); see below): each field sees the others at
the start of the step, so the result does not depend on the order. A steady field (one without a
time derivative) coupled to a transient field through such sources, e.g. a Poisson potential
$-\nabla^2 \phi = s c$ with $\partial_t c = D \nabla^2 c + k \phi$, is computed from $c^n$, and the
step of $c$ uses $\phi^n$, so $\phi$ lags $c$ by one step. Its initial condition should be the
steady solution of the initial $c$, which the first step of $c$ uses.

The solve stops when the unpreconditioned residual satisfies
$\|b - A u\|_2 \le$ [!param](/TensorSolver/ImplicitFDSolver/l_tol) $\|b\|_2$, starting from the
previous solution. When a solve does not converge within
[!param](/TensorSolver/ImplicitFDSolver/l_max_its) iterations, a
`TensorSolveIterationAdaptiveDT` time stepper retries the time step with a smaller time step;
with any other time stepper it is an error. After the solve, the
operators' output buffers hold the operators applied to the new solution.

The [!param](/TensorSolver/ImplicitFDSolver/root_compute) is evaluated before every solve (with
[!param](/TensorSolver/ImplicitFDSolver/nonlinear_solve): at every iterate) to update the explicit
terms, such as the source. The field operators in it are skipped, since the
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

## Nonlinear iteration

With `nonlinear_solve = true` each step is iterated to the fully implicit step of all fields
(backward Euler, or BDF2 with the time derivatives' `order = 2`). `nonlinear_solver` picks
the driver independently of `linear_solver`: `TORCH` is a
plain Picard iteration in torch, `PETSC` a PETSc SNES (options prefix `nonlinear_`), and `AUTO`
selects `PETSC` with `linear_solver = PETSC` or when nonlinear options are given, `TORCH`
otherwise. Options starting with `-snes` or `-npc_` in `petsc_options` go to the SNES; the other
options configure the field solves and are an error unless `linear_solver = PETSC`. Empty option
lists count as no options.

The SNES can drive the torch field solves, e.g. Anderson mixing on MPS. Its vectors then live on
the host (also on CUDA when PETSc is built without CUDA; with a CUDA PETSc they stay on the
device), so every residual, operator product and preconditioner application copies the iterate
between host and device, in double precision on the host and the device precision on the device.
This works but is slow, and the solver warns at the first solve; `nonlinear_solver = TORCH` keeps
the iteration on the device.

On the SNES the default is the
Picard iteration, with the field solves as its preconditioner. It converges only while the
explicit terms vary slowly enough with the fields, roughly $\Delta t\, L f'' < 1$ for an
Allen-Cahn reaction. Anderson mixing or nonlinear GMRES extend that several times. They must take
the field solves as their nonlinear preconditioner:

```
petsc_options_iname = '-snes_type -npc_snes_type'
petsc_options_value = 'anderson newtonls'
```

Without `-npc_snes_type newtonls` these methods (and `nrichardson`) do not use the field solves
at all: they mix the raw residuals (0 linear iterations). On a stiff problem they then fail to
converge. Their iterates also need not keep the zero-mean right-hand side of a field with a
singular operator: the residual evaluation then stops with the error on the mean of that field.

During the iteration the explicit computes run at every iterate and at the new time
$t^{n+1}$, so the converged step is $A U = \sum_i r_i + f(U, t^{n+1})$, and boundary values
taken from buffers are refilled from the iterate. The iteration stops when
$\|F(U)\|_2 <$ [!param](/TensorSolver/ImplicitFDSolver/nl_abs_tol) or
$\|F(U)\|_2 \le$ [!param](/TensorSolver/ImplicitFDSolver/nl_rel_tol) $\|F(U^0)\|_2$, and fails
after [!param](/TensorSolver/ImplicitFDSolver/nl_max_its) iterations (`-snes_rtol`, `-snes_atol`
and `-snes_max_it` in `petsc_options` override these for the SNES). A failed step is retried with
a smaller time step by `TensorSolveIterationAdaptiveDT`, starting again from $u^n$; with any other
time stepper it is an error. The iteration count reported to the time stepper is the number of
nonlinear iterations, so its `min_iterations` and `max_iterations` refer to them; the total of
linear iterations is printed with it.

## Boundary conditions

On a non-periodic axis, each face of $u$ has a [TensorDirichletBC.md], [TensorNeumannBC.md] or
[TensorRobinBC.md] (see the [BCs system](syntax/TensorComputes/BCs/index.md)), with values taken at
the new time for functions and at the start of the time step for buffers (at the iterate with
[!param](/TensorSolver/ImplicitFDSolver/nonlinear_solve)). With boundary values the
operator action is affine, $A u + g$, and the solver solves $A u = b - g$, with $g$ the action on
$u = 0$, recomputed every solve. A Dirichlet face, or a Robin face with $a \ne 0$, makes the system
definite; with Neumann faces only it keeps the constants as its nullspace, and the right-hand
side, including the fluxes, must then have zero mean up to round-off. Source and fluxes that
balance in the continuous problem generally leave a discrete mean of order $h^2$, which is an
error: the data must balance on the grid. Supported are:

- the time derivative;
- the Laplacian, with Dirichlet, Neumann and Robin faces for the 3-point stencil and Dirichlet
  and Neumann faces for the 5-point stencil, whose second ghost layer makes a Robin operator
  nonsymmetric;
- [FiniteDifferenceDivGrad.md], with Dirichlet, Neumann and Robin faces;
- the 3-point [FiniteDifferenceBiharmonic.md], with Dirichlet faces ($u = g$, $\nabla^2 u = 0$)
  and Neumann faces ($\partial u/\partial n = q$, $\partial(\nabla^2 u)/\partial n = 0$).

A Robin face with $a b < 0$ is an error: its ghost term $a / (a/2 + b/\Delta x)$ turns negative
once $\Delta x < 2 |b / a|$, after which the operator may be indefinite, depending on the rest of
the problem, so whether it could be solved would depend on the grid.

## Variable coefficients

An operator with a coefficient per cell ([FiniteDifferenceDivGrad.md]) gives the solver one
stencil field per offset. The torch Jacobi preconditioner and the smoothers use the cell-wise
diagonal, which varies with the coefficient. On the PETSc coarse levels the coefficient is the
mean over each coarse cell's children, and the operator is rediscretised from it. A cell whose
diagonal is not positive (a coefficient that is zero on every face of a cell, without a time
derivative) is an error. The coarse levels are reassembled when the coefficient changes.

On the coarse multigrid levels, a stencil entry that reaches a ghost cell is folded onto the
mirrored interior cell, times the ghost factor $-(a/2 - b/s)/(a/2 + b/s)$ of the homogeneous
condition with $s = (2l+1) h$ for ghost layer $l$ and level spacing $h$ ($-1$ for Dirichlet, $1$
for Neumann). The interpolation folds the same way. This affects only the preconditioner.

## Example Input File Syntax

!listing test/tests/implicit_fd/be_mode.i block=TensorSolver

With boundary conditions whose values vary along the faces:

!listing test/tests/implicit_fd/bc_mms.i block=TensorComputes/BCs

With a variable coefficient:

!listing test/tests/implicit_fd/divgrad_mms.i block=TensorComputes/Solve TensorSolver

!syntax parameters /TensorSolver/ImplicitFDSolver

!syntax inputs /TensorSolver/ImplicitFDSolver

!syntax children /TensorSolver/ImplicitFDSolver
