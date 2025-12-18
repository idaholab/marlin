# LBMComputeForces

!syntax description /TensorComputes/Solve/LBMComputeForces

LBMComputeForces calculates forces for LBM simulations. Currently available forces are gravity and buoyancy.

## Overview

Computes body force contributions (e.g., gravity or external driving force) required by the
LBM forcing term. For gravity, set [!param](/TensorComputes/Solve/LBMComputeForces/enable_gravity) to `true` and supply gravitational acceleration via [!param](/TensorComputes/Solve/LBMComputeForces/gravity). Please note that gravity is applied in `y` direction by default, but it can be changed to `x` or `z` by setting [!param](/TensorComputes/Solve/LBMComputeForces/gravity_direction) to `0` or `2` respectively. When coupling heat and mass transfer to enable Boussinesq approximation of buoyancy, set [!param](/TensorComputes/Solve/LBMComputeForces/enable_buoyancy) to `true` and supply temperature buffer via [!param](/TensorComputes/Solve/LBMComputeForces/temperature), reference temperature via [!param](/TensorComputes/Solve/LBMComputeForces/T0) and reference density via [!param](/TensorComputes/Solve/LBMComputeForces/rho0).
output force buffer via [!param](/TensorComputes/Solve/LBMComputeForces/buffer).

### Gravity

When `enable_gravity = true`, a constant gravitational force is applied:

$$
\mathbf{F}_g = \rho \mathbf{g}
$$

where $\rho$ is the local density and $\mathbf{g}$ is the gravitational acceleration vector defined by its magnitude (`gravity`) and direction (`gravity_direction`).

### Buoyancy

When `enable_buoyancy = true`, a buoyancy force is applied using the Boussinesq approximation:

$$
\mathbf{F}_b = \rho_0 \mathbf{g} (T - T_0)
$$

where:
- $\rho_0$ is the reference density (`rho0`).
- $T$ is the local temperature.
- $T_0$ is the reference temperature (`T0`).

> **Note**: This object is currently under active development. Additional force contributions (e.g., surface tension for multiphase flows) are being added to support more complex physics.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [Apply_forces]
      type = LBMApplyForces
      buffer = fpc
      velocity = velocity
      rho = density
      forces = F
      tau0 = 1.0
    []
  []
[]
!listing-end


!syntax parameters /TensorComputes/Solve/LBMComputeForces

!syntax inputs /TensorComputes/Solve/LBMComputeForces

!syntax children /TensorComputes/Solve/LBMComputeForces
