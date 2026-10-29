# CoefficientTimeDerivativeOperator

!syntax description /TensorComputes/Solve/CoefficientTimeDerivativeOperator

The time derivative term $a \, \partial u / \partial t$ of an
[ImplicitFDSolver](ImplicitFDSolver.md) field, with a cell-centred coefficient buffer
$a \geq 0$. It is [TimeDerivativeOperator](TimeDerivativeOperator.md) with each row multiplied
by $a$: with the weights $w_0, w_1, w_2$ of the step (backward Euler, or variable-step BDF2 with
`order = 2`) the action is $a \, w_0 u / \Delta t$ and the right-hand side contribution is
$-a (w_1 u^n + w_2 u^{n-1}) / \Delta t$.

An example is the domain parameter $\psi$ of the smoothed boundary method, where the
$\psi$-multiplied equation $\psi \, \partial c / \partial t = \nabla \cdot (\psi M \nabla \mu)$
keeps the operator symmetric and conserves $\sum_i \psi_i c_i$. Other uses are a heat capacity
$\rho c_p$, or a susceptibility $\partial c / \partial \mu$ in a grand-potential model.

The coefficient is read when the step's right-hand side is formed, and must not change within the
step (a nonlinear iteration that rewrites $a$ with a different value is an error). The time
derivative of a time-dependent $a$, the term $u \, \partial a / \partial t$ of
$\partial (a u) / \partial t$, is not included. A negative $a$ is an error. In the multigrid
preconditioner of the PETSc solver a coarse cell's $a$ is the mean over its children.

## Example Input File Syntax

!listing test/tests/implicit_fd/coefficient_dt.i block=TensorComputes/Solve

!syntax parameters /TensorComputes/Solve/CoefficientTimeDerivativeOperator

!syntax inputs /TensorComputes/Solve/CoefficientTimeDerivativeOperator

!syntax children /TensorComputes/Solve/CoefficientTimeDerivativeOperator
