# LBMForceDistribution

!syntax description /TensorComputes/Solve/LBMForceDistribution

This compute object adds the force distribution source term to the hydrodynamic distribution function for phase field lattice Boltzmann simulations.

## Overview

Computes the Guo\-style force distribution that incorporates body forces and density\-contrast
terms into the collision step. Supply the gradient of the phase field via
[!param](/TensorComputes/Solve/LBMForceDistribution/grad_phi), the velocity via
[!param](/TensorComputes/Solve/LBMForceDistribution/velocity), and the force vector via
[!param](/TensorComputes/Solve/LBMForceDistribution/forces). Liquid and gas densities are set with
[!param](/TensorComputes/Solve/LBMForceDistribution/rho_l) and
[!param](/TensorComputes/Solve/LBMForceDistribution/rho_g). For spatially varying relaxation,
enable [!param](/TensorComputes/Solve/LBMForceDistribution/is_dynamic_relaxation) and provide
[!param](/TensorComputes/Solve/LBMForceDistribution/tau_tensor).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/apply_forces_hydro

!syntax parameters /TensorComputes/Solve/LBMForceDistribution

!syntax inputs /TensorComputes/Solve/LBMForceDistribution

!syntax children /TensorComputes/Solve/LBMForceDistribution
