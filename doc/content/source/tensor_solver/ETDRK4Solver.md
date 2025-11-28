# ETDRK4Solver

!syntax description /TensorSolver/ETDRK4Solver

Fourth-order exponential time differencing integrator for split-operator problems. The solver
applies the Runge-Kutta ETD scheme of Cox and Matthews using precomputed exponential
coefficients to evolve the linear reciprocal term exactly over each substep while integrating
nonlinear contributions in spectral space. The optional
[!param](/TensorSolver/ETDRK4Solver/substeps) parameter subdivides each transient step to keep
high-order stability when the full time step is large compared to the stiff linear dynamics.

## Example Input File Syntax

!listing test/tests/solvers/etdrk4_diffusion.i block=TensorSolver

!syntax parameters /TensorSolver/ETDRK4Solver

!syntax inputs /TensorSolver/ETDRK4Solver

!syntax children /TensorSolver/ETDRK4Solver
