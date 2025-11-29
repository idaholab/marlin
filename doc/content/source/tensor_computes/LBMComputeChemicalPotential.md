# LBMComputeChemicalPotential

!syntax description /TensorComputes/Solve/LBMComputeChemicalPotential

This compute object computes chemical potential from parabiolic free energy eqution for lattice Boltzmann simulations.

## Overview

Evaluates the Cahn\-Hilliard chemical potential for a phase field $\phi$ using a double\-well
potential and interfacial energy term. Provide the scalar field via
[!param](/TensorComputes/Solve/LBMComputeChemicalPotential/phi) and its Laplacian via
[!param](/TensorComputes/Solve/LBMComputeChemicalPotential/laplacian_phi). Control the interface
thickness with [!param](/TensorComputes/Solve/LBMComputeChemicalPotential/thickness) and the
surface tension with [!param](/TensorComputes/Solve/LBMComputeChemicalPotential/sigma).

The chemical potential $\mu$ is computed as:

$$
\mu = \frac{\sigma}{D} \phi (\phi - 1) - D \sigma \nabla^2 \phi
$$

where:
- $\phi$ is the phase field order parameter.
- $\nabla^2 \phi$ is the Laplacian of the order parameter.
- $D$ is the interface thickness (`thickness`).
- $\sigma$ is the interfacial tension coefficient (`sigma`).

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [potential]
      type = LBMComputeChemicalPotential
      buffer = mu
      phi = phi
      laplacian_phi = laplacian_phi
      thickness = D
      sigma = sigma
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMComputeChemicalPotential

!syntax inputs /TensorComputes/Solve/LBMComputeChemicalPotential

!syntax children /TensorComputes/Solve/LBMComputeChemicalPotential
