# LBMIsotropicLaplacian

!syntax description /TensorComputes/Solve/LBMIsotropicLaplacian

This object uses isotropic finite difference method otot compute the Lapcian of scalar field for LBM simulations.

## Overview

Computes an isotropic finite\-difference approximation to the Laplacian $\nabla^2 \phi$ on the
LBM grid. Provide the scalar field with
[!param](/TensorComputes/Solve/LBMIsotropicLaplacian/scalar_field) and select the destination
scalar buffer via [!param](/TensorComputes/Solve/LBMIsotropicLaplacian/buffer).

The isotropic Laplacian $\nabla^2 \phi$ is computed using a weighted finite difference scheme:

$$
\nabla^2 \phi (\mathbf{x}) = \frac{2}{c_s^2} \sum_i w_i (\phi(\mathbf{x} + \mathbf{e}_i) - \phi(\mathbf{x}))
$$

where:
- $w_i$ are the lattice weights.
- $\mathbf{e}_i$ are the discrete lattice velocities.
- $c_s$ is the lattice speed of sound.

This operation is implemented as a convolution with a kernel constructed from the lattice weights.

> **Note**: This method requires an isotropic lattice stencil (e.g., D2Q9). It is not compatible with stencils that lack sufficient isotropy for Laplacian calculations, such as D3Q19.

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
    [laplacian_phi]
      type = LBMIsotropicLaplacian
      buffer = laplacian_phi
      scalar_field = phi
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMIsotropicLaplacian

!syntax inputs /TensorComputes/Solve/LBMIsotropicLaplacian

!syntax children /TensorComputes/Solve/LBMIsotropicLaplacian
