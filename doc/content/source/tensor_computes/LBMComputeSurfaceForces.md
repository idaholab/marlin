# LBMComputeSurfaceForces

!syntax description /TensorComputes/Solve/LBMComputeSurfaceForces

This object acts similarly to LBMComputeForces and computes surface forces from chemical potential and gradient of phase-field order parameter.

## Overview

Computes interfacial tension force density from a chemical potential and gradient of phase field parameter for phase\-field LBM.
coupling. Supply the chemical potential with
[!param](/TensorComputes/Solve/LBMComputeSurfaceForces/chemical_potential) and the gradient with
[!param](/TensorComputes/Solve/LBMComputeSurfaceForces/grad_phi). The result is a vector field
written to [!param](/TensorComputes/Solve/LBMComputeSurfaceForces/buffer).

The surface force density $\mathbf{F}_s$ is computed as:

$$
\mathbf{F}_s = \mu \nabla \phi
$$

where:
- $\mu$ is the chemical potential (`chemical_potential`).
- $\nabla \phi$ is the gradient of the phase field order parameter (`grad_phi`).

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [forces]
      type = LBMComputeSurfaceForces
      buffer = forces
      chemical_potential = mu
      grad_phi = grad_phi
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMComputeSurfaceForces

!syntax inputs /TensorComputes/Solve/LBMComputeSurfaceForces

!syntax children /TensorComputes/Solve/LBMComputeSurfaceForces
