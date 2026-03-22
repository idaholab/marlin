# LBMComputeEffectiveRelaxation

!syntax description /TensorComputes/Initialize/LBMComputeEffectiveRelaxation

This compute object calculates a spatially varying effective relaxation time matrix for rarefied gas Lattice Boltzmann simulations. It accounts for slip and transitional flow regime effects through the local pore size and Knudsen number fields. Currently limited to D2Q9 stencils.

## Overview

For rarefied gas flows the standard single relaxation time is insufficient because the mean free path of gas molecules becomes comparable to the pore size. This object computes a per-node diagonal relaxation matrix with three distinct relaxation times for the 9 moment modes of D2Q9 MRT collision:

### Shear relaxation ($\tau_s$)

Controls viscous stress relaxation with a slip-corrected form:

$$ \tau_s = \frac{1}{2} + \sqrt{\frac{6}{\pi}} \frac{d_p \cdot \text{Kn}}{1 + 2\,\text{Kn}} $$

where $d_p$ is the local pore size (in lattice units) and $\text{Kn}$ is the local Knudsen number.

### Energy flux relaxation ($\tau_q$)

Derived from second-order slip boundary theory:

$$ \tau_q = \frac{1}{2} + \frac{3 + \pi \, A_2 \, (2\tau_s - 1)^2}{8\,(2\tau_s - 1)} $$

where $A_2$ is the second-order slip coefficient (default 0.8).

### Diffusion relaxation ($\tau_d$)

Based on the ratio of mean free path to the effective pore spacing:

$$ \tau_d = \frac{1}{2} + \frac{3\sqrt{3}}{8} \frac{\lambda}{dx \, (1 + 2\,\text{Kn})} $$

where $\lambda$ is the molecular mean free path and $dx$ is the physical grid spacing.

### Relaxation matrix layout

The output buffer has shape `(Nx, Ny, 1, 9)` and stores the **inverse** of each relaxation time (i.e. $1/\tau$) for compatibility with the MRT collision operator. The 9 entries per node are:

| Index | Mode | Value |
|-------|------|-------|
| 0 | Conserved (density) | 1 |
| 1 | Conserved | 1/1.1 |
| 2 | Conserved | 1/1.2 |
| 3 | Diffusion | $1/\tau_d$ |
| 4 | Energy flux | $1/\tau_q$ |
| 5 | Diffusion | $1/\tau_d$ |
| 6 | Energy flux | $1/\tau_q$ |
| 7 | Shear stress | $1/\tau_s$ |
| 8 | Shear stress | $1/\tau_s$ |

### Requirements

- A `binary_media` buffer must be defined in `LatticeBoltzmannProblem` (solid cells are zeroed out).
- Only D2Q9 stencils are currently supported.
- Input buffers `local_pore_size` and `local_Knudsen_number` must be provided (typically loaded from HDF5 files).

## Example Input File Syntax

!listing examples/lbm/rarefied_gas/channel.i block=TensorComputes/Initialize/relaxation_matrix_init

The relaxation matrix is then passed to the MRT collision operator:

!listing examples/lbm/rarefied_gas/channel.i block=TensorComputes/Solve/collision

!syntax parameters /TensorComputes/Initialize/LBMComputeEffectiveRelaxation

!syntax inputs /TensorComputes/Initialize/LBMComputeEffectiveRelaxation

!syntax children /TensorComputes/Initialize/LBMComputeEffectiveRelaxation
