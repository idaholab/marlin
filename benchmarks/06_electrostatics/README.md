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
| MEMPHIS, 100^2                    | 171.50 |  137.50 |  120.00 |  103.00 |
| this input, 100^2, solve dt 0.5   | 172.90 |  138.40 |  121.83 |  104.00 |
| this input, 200^2, solve dt 0.5   | 172.85 |  138.50 |  121.88 |  104.01 |
| this input, 100^2, solve dt 0.05  | 172.90 |  138.37 |  121.70 |  103.82 |

The 100^2 and 200^2 grids agree to 0.014 at t = 400. Against a solve step of 0.05, F at t = 400
differs by 0.176, 0.081 and 0.019 at 0.5, 0.25 and 0.1 (about first order; backward Euler with
the explicit terms lagged). On the cuts at t = 400, c differs by up to 1.9e-2 between solve steps
0.5 and 0.05 and by 7e-3 between 100^2 and 200^2. The value
at t = 0 is that of the initial guess of `phi`, solved from the first step on. Wall time to
t = 400 on one CPU core: 20 s (100^2), 67 s (200^2).

`plot_comparison.py <run_dir>` draws `free_energy_6a.png` (the curves above) and `cost_6a.png`
(memory against wall time per unit simulated time) from the three runs in `<run_dir>` and the
PFHub submissions' data, which it downloads. The PFHub costs are as their submitters reported
them, on 2017 hardware with 1 to 16 cores; three of them ran to t = 1000. Our runs are one core
of an Apple M5 Pro, with memory the whole process as MOOSE reports it; one 200^2 field is
0.3 MB, so the fields are a small part of it.
