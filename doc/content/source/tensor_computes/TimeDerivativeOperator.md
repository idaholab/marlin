# TimeDerivativeOperator

!syntax description /TensorComputes/Solve/TimeDerivativeOperator

## Overview

The time derivative term for implicit finite-difference solves, backward Euler
([!param](/TensorComputes/Solve/TimeDerivativeOperator/order) $= 1$) or BDF2 ($= 2$).
With backward Euler its action on the solved field $u$ is $u/\Delta t$, and it contributes
$u^n/\Delta t$ to the right-hand side, with $\Delta t$ the current time step.
Listed together with the spatial operators of a field, an implicit solver
then solves

!equation
\frac{u - u^n}{\Delta t} + \sum_i L_i u = f.

BDF2 uses the variable-step weights (as MOOSE's `BDF2` time integrator), with
$\Delta t_\text{old}$ the previous accepted step and $s = \Delta t + \Delta t_\text{old}$,

!equation
\frac{w_0 u + w_1 u^n + w_2 u^{n-1}}{\Delta t} + \sum_i L_i u = f, \quad
w_0 = 1 + \frac{\Delta t}{s}, \quad w_1 = -\frac{s}{\Delta t_\text{old}}, \quad
w_2 = \frac{\Delta t^2}{\Delta t_\text{old}\, s},

and backward Euler on the first step, which has no $u^{n-1}$. The weights sum to zero, so the
time derivative of a constant vanishes and a conserved total stays conserved. A step that is
retried with a smaller $\Delta t$ keeps $u^n$, $u^{n-1}$ and $\Delta t_\text{old}$. BDF2 is
second order in time when the whole step is implicit: with the
[!param](/TensorSolver/ImplicitFDSolver/nonlinear_solve) iteration, or when every term is one of
the field's operators. With explicit terms taken from the start of the step (the lagged scheme
without `nonlinear_solve`), the step is first order overall. BDF2 is zero-stable for step ratios
$\Delta t / \Delta t_\text{old} < 1 + \sqrt{2}$.

The operator is positive definite, and its stencil is the single centre entry
$w_0/\Delta t$ at every grid spacing ($w_0 = 1$ for backward Euler). With BDF2, $w_0$ changes
with the step ratio, so the multigrid coarse levels are rebuilt whenever either $\Delta t$ or
the ratio changes.

!syntax parameters /TensorComputes/Solve/TimeDerivativeOperator

!syntax inputs /TensorComputes/Solve/TimeDerivativeOperator

!syntax children /TensorComputes/Solve/TimeDerivativeOperator
