# Implicit FD Cahn-Hilliard size scaling

Single-device timing of the implicit finite-difference solve of
`../01_spinodal_decomposition/1a_implicit_fd.i` (backward Euler, biharmonic
implicit, double-well flux divergence explicit) for the `PETSC` (CG +
geometric multigrid) and `TORCH_PCG` (Jacobi CG) linear solvers, on grids from
512^2 to 8192^2 cells.

The domain stays [0,200]^2 and only the grid is refined, so the runs are not
PFHub 1a results; they are for timing only. Each time step is 1000 linear
solves (`substeps`), and the `wall_time` postprocessor gives the run time.

```
./run_scaling.sh
SIZES="1024 2048" BACKENDS=PETSC ./run_scaling.sh
DEVICE=cpu ./run_scaling.sh     # PETSc and torch on a Mac CPU
```

The device defaults to `cuda` when `nvidia-smi` is present, `mps` on macOS and
`cpu` otherwise. On MPS (single precision, no PETSc) only `TORCH_PCG` runs, with
`l_tol = 1e-5`, so those timings are not like-for-like with the double precision
runs. The time per step excludes the first step, which carries the startup and
solver setup.

Results go to `results/` (`summary.csv`, one log and postprocessor CSV per
run, `provenance.txt`).
