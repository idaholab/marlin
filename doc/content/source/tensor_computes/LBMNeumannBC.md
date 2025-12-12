# LBMNeumannBC

!syntax description /TensorComputes/Boundary/LBMNeumannBC

Implements a Neumann boundary condition for Lattice Boltzmann simulations. It enforces a fixed gradient of a microsocpic variable at the boundary.

## Overview

This boundary condition enforces a specified gradient at the boundaries. It calculates the required macroscopic value at the boundary to satisfy the gradient condition and then uses the Dirichlet implementation logic to apply it. Note that the gradient specified at the boundary is not the gradient of macroscopic variable. Rather, it is the gradient of distribution functions (microscopic variable). This boundary condition only computes missing directions by applying Non-Equilibrium Extrapolation Method (NEEM). This means that the non-equilibirum part of the distribution function is computed from existing values in the current cell and the equilibrium part is computed from the prescribed gradients at the boundary.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` and `regional` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. The `regional` boundary enables users to mark different walls of the domain with different 
values and apply Neumann condition only in those regions.

The update rule for the distribution function at the boundary node $\mathbf{x}_b$ is:

$$
f_i^{new}(\mathbf{x}_b) = f_i^{eq}(\rho(\mathbf{x}_b) + G, \mathbf{u}) + f_i^{neq}(\mathbf{x}_b)
$$

where:
- $G$ is the specified gradient parameter (`gradient`).
- $\rho(\mathbf{x}_b)$ is the current macroscopic density (or scalar) at the boundary.
- $f_i^{neq}(\mathbf{x}_b) = f_i^{old}(\mathbf{x}_b) - f_i^{eq}(\rho(\mathbf{x}_b), \mathbf{u})$ is the non-equilibrium part from the previous state.

Assuming the equilibrium distribution is linear with respect to the scalar field $\rho$, this simplifies to adding a source term proportional to $G$:

$$
f_i^{new}(\mathbf{x}_b) = f_i^{old}(\mathbf{x}_b) + f_i^{eq}(G, \mathbf{u})
$$

This effectively injects a population flux corresponding to the macroscopic value $G$ into the boundary cells.

## Example Input File Syntax

!listing
[TensorComputes]
  [Boundary]
    [heat_source]
      type = LBMNeumannBC
      buffer = g
      f_old = gpc
      feq=geq
      velocity = velocity
      rho = T
      gradient = 0.001
      region_id = 3
      boundary = regional
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Boundary/LBMNeumannBC

!syntax inputs /TensorComputes/Boundary/LBMNeumannBC

!syntax children /TensorComputes/Boundary/LBMNeumannBC
