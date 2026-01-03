# LBMIsotropicGradient

!syntax description /TensorComputes/Solve/LBMIsotropicGradient

This object uses isotropic finite difference method to compute the spatial gradient of scalar field for LBM simulations.

## Overview

Computes an isotropic finite\-difference approximation to $\nabla \phi$ on the LBM grid. Provide
the scalar field with
[!param](/TensorComputes/Solve/LBMIsotropicGradient/scalar_field) and select the destination
vector buffer via [!param](/TensorComputes/Solve/LBMIsotropicGradient/buffer).

The isotropic gradient $\nabla \phi$ is computed using a weighted finite difference scheme:

$$
\nabla \phi (\mathbf{x}) = \frac{1}{c_s^2} \sum_i w_i \phi(\mathbf{x} + \mathbf{e}_i) \mathbf{e}_i
$$

where:
- $w_i$ are the lattice weights.
- $\mathbf{e}_i$ are the discrete lattice velocities.
- $c_s$ is the lattice speed of sound.

This operation is implemented as a convolution with a kernel constructed from the lattice weights and velocities.

> **Note**: This method requires an isotropic lattice stencil (e.g., D2Q9 or D3Q27). It is not compatible with stencils that lack sufficient isotropy for gradient calculations, such as D3Q19.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [compute_phi]
      type = LBMComputeDensity
      buffer = phi
      f = h
    []
    [grad_phi]
      type = LBMIsotropicGradient
      buffer = grad_phi
      scalar_field = phi
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMIsotropicGradient

!syntax inputs /TensorComputes/Solve/LBMIsotropicGradient

!syntax children /TensorComputes/Solve/LBMIsotropicGradient
