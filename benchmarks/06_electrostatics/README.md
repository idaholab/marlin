# 6a - Spinodal Decomposition with Electrostatics

`6a_implicit_fd.i` solves PFHub benchmark 6a (square domain) with `ImplicitFDSolver`: the
transient concentration `c` (backward Euler, the biharmonic implicit) and the steady potential
`phi` in one solver, every explicit term from the start of the step. `cut_planes.py <file_base>`
writes the PFHub outputs (`free_energy`, `x_cut_plane`, `y_cut_plane`) from a run's CSV and HDF5
output.

Free energy against the PFHub submissions (CPU, PETSc, 2026-10-26, commit of this input):

| run                               |  t = 5 |  t = 50 | t = 200 | t = 400 |
| --------------------------------- | ------:| -------:| -------:| -------:|
| INL MOOSE, 160^2                  | 170.47 |  138.46 |  121.78 |  103.61 |
| CORAL (MOOSE)                     | 173.00 |  138.63 |  121.32 |  104.87 |
| PRISMS-PF                         | 172.56 |  140.97 |  124.32 |  105.75 |
| FiPy, 100^2                       | 171.29 |  138.93 |  126.07 |  114.28 |
| MMSP, dx = 1                      | 173.98 |  134.48 |  117.34 |  102.89 |
| MEMPHIS, 100^2 (a)                | 171.50 |  137.50 |  120.00 |  103.00 |
| this input, 100^2, solve dt 0.5   | 172.90 |  138.40 |  121.83 |  104.00 |
| this input, 200^2, solve dt 0.5   | 172.85 |  138.50 |  121.88 |  104.01 |
| this input, 100^2, solve dt 0.05  | 172.90 |  138.35 |  121.70 |  103.82 |

The 100^2 and 200^2 grids agree to 0.014 at t = 400. Against a solve step of 0.05, F at t = 400
differs by 0.176, 0.081 and 0.019 at 0.5, 0.25 and 0.1 (about first order; backward Euler with
the explicit terms lagged). On the cuts at t = 400, c differs by up to 1.9e-2 between solve steps
0.5 and 0.05 and by 7e-3 between 100^2 and 200^2. The value at t = 0 is that of the initial
guess of `phi`, solved from the first step on (left out of the PFHub file). Wall time to t = 400
on one CPU core: 18 s (100^2), 65 s (200^2), from MOOSE's "Finished Executing".

(a) MEMPHIS reports F at t = 4, 6, 40 and 60; its t = 5 and t = 50 entries are linear
interpolations.

`6a_implicit_fd_nonlinear.i` is the fully implicit version (`nonlinear_solve = true`): the
nonlinear solve iterates each step to backward Euler of c and phi together, with f'(c) and the
coupling at t^{n+1}, by Anderson mixing over the field solves. The time step follows the
nonlinear iterations (`TensorSolveIterationAdaptiveDT`, window 8 to 15), capped at
`dtmax = 0.05`. CPU, PETSc, 100^2, 2026-10-27:

| run                                   |  t = 5 |  t = 50 | t = 200 | t = 400 |
| ------------------------------------- | ------:| -------:| -------:| -------:|
| nonlinear, dtmax 0.05                 | 172.80 |  138.32 |  121.68 |  103.79 |
| nonlinear, dtmax 0.005 (reference)    | 172.86 |  138.33 |         |         |

It took 8030 steps to t = 400 with no cuts, 8.9 nonlinear iterations per step on average and
at most 10. Wall time to t = 400 on one CPU core: 1032 s, against 18 s for the lagged input at
its larger steps. The XDMF/HDF5 output is written at every step (1.2 GB at 100^2).

Error of F against the reference, with the solve dt at that time (100^2):

| run                              | t = 5           | t = 50          |
| -------------------------------- | ---------------:| ---------------:|
| lagged, solve dt up to 0.5       | +0.039 (0.023)  | +0.078 (0.25)   |
| lagged, solve dt up to 0.1       | +0.039 (0.023)  | +0.046 (0.1)    |
| nonlinear, dtmax 0.02            | -0.020 (0.02)   | -0.002 (0.02)   |
| nonlinear, dtmax 0.05            | -0.060 (0.05)   | -0.005 (0.05)   |
| nonlinear, dtmax 0.1             | -0.123 (0.1)    | -0.009 (0.1)    |
| nonlinear, no cap                | -0.179 (0.15)   | -0.013 (0.15)   |
| nonlinear, solve dt up to 0.5 (a)| -0.009 (0.023)  | +0.002 (0.25)   |

At the same solve dt the nonlinear error is several times smaller: at t = 5, -0.009 against
+0.039 at dt = 0.023, and at t = 50, +0.002 against +0.078 at 0.25. The errors are first order
in dt (dtmax 0.02, 0.05, 0.1: -0.020, -0.060, -0.123 at t = 5). Up to t = 5 every lagged run is
still at the same steps (its step grows from 1e-4), so they agree there. The cap of 0.05 is a
demonstration setting: its steps are about twice the lagged input's at t = 5 and a fifth of
them at t = 50.

(a) `6a_implicit_fd.i` with `TensorSolver/nonlinear_solve=true` and the same Anderson options, on
the lagged input's step schedule (IterationAdaptiveDT, 20 solves per step). It reaches t = 400
(F = 103.77) with up to 24 iterations per step. The plain Picard iteration (without Anderson)
fails there near t = 41, where the solve dt passes about 0.19.

`plot_comparison.py <run_dir>` draws `free_energy_6a.png` (the curves above) and `cost_6a.png`
(memory against wall time per unit simulated time) from the three runs in `<run_dir>` and the
PFHub submissions' data, which it downloads. The PFHub costs are as their submitters reported
them, on 2017 hardware (MEMPHIS 2022) with 1 to 16 cores; three of them ran to t = 1000. Our
runs are one core of an Apple M5 Pro, with memory the whole process as MOOSE reports it; one 200^2 field is
0.3 MB, so the fields are a small part of it.
