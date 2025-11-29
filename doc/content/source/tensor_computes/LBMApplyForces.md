# LBMApplyForces

!syntax description /TensorComputes/Solve/LBMApplyForces

LBMApplyForces adds forces onto LBM distribution fucntion. The forces act as source term.

## Overview

Applies a body force term to a post\-collision distribution using the Guo forcing scheme. Provide
the density via [!param](/TensorComputes/Solve/LBMApplyForces/rho) and the force vector via
[!param](/TensorComputes/Solve/LBMApplyForces/forces). The relaxation time is referenced via
[!param](/TensorComputes/Solve/LBMApplyForces/tau0).

The force is applied as a source term $S_i$ to the distribution function. The update rule is:

$$
f_i = f_i + \left( 1 - \frac{1}{2\tau} \right) S_i
$$

Currently, a first-order approximation is implemented for the source term $S_i$:

$$
S_i = w_i \rho \frac{\mathbf{e}_i \cdot \mathbf{F}}{c_s^2}
$$

where:
- $w_i$ are the lattice weights.
- $\rho$ is the macroscopic density.
- $\mathbf{e}_i$ are the discrete lattice velocities.
- $\mathbf{F}$ is the external force vector.
- $c_s$ is the lattice speed of sound.

> **Note**: The full second-order Guo forcing scheme (which includes terms involving macroscopic velocity $\mathbf{u}$) is currently a work in progress.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [Compute_forces]
      type = LBMComputeForces
      buffer = F
      rho0 = 1.0
      temperature = T
      T0 = 1.0
      enable_buoyancy = true
      gravity = g
      gravity_direction = 2
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMApplyForces

!syntax inputs /TensorComputes/Solve/LBMApplyForces

!syntax children /TensorComputes/Solve/LBMApplyForces
