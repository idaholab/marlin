# LBMComputeForces

!syntax description /TensorComputes/Solve/LBMComputeForces

LBMComputeForces calculates forces for LBM simulations. Currently available forces are gravity and buoyancy.

## Overview

Computes body force contributions (e.g., gravity or external driving force) required by the
LBM forcing term. For gravity, set [!param](/TensorComputes/Solve/LBMComputeForces/enable_gravity) to `true` and supply gravitational acceleration via [!param](/TensorComputes/Solve/LBMComputeForces/gravity). Please not that gravity is applied in `y` direction by default, but it can be changed to `x` or `z` by setting [!param](/TensorComputes/Solve/LBMComputeForces/gravity_direction) to `0` or `2` respectively. When coupling heat and mass transfer to enable Boussinesq approximation of buoyancy, set [!param](/TensorComputes/Solve/LBMComputeForces/enable_buoyancy) to `true` and supply temperature buffer via [!param](/TensorComputes/Solve/LBMComputeForces/temperature), reference temperature via [!param](/TensorComputes/Solve/LBMComputeForces/T0) and reference density via [!param](/TensorComputes/Solve/LBMComputeForces/rho0).
output force buffer via [!param](/TensorComputes/Solve/LBMComputeForces/buffer).

## Example Input File Syntax

!listing examples/lbm/natural_convection_3D/convection.i block=TensorComputes/Solve/Compute_forces

!syntax parameters /TensorComputes/Solve/LBMComputeForces

!syntax inputs /TensorComputes/Solve/LBMComputeForces

!syntax children /TensorComputes/Solve/LBMComputeForces
