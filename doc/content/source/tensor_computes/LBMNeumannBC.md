# LBMNeumannBC

!syntax description /TensorComputes/Boundary/LBMNeumannBC

Implements a Neumann boundary condition for Lattice Boltzmann simulations. It enforces a fixed gradient of a microsocpic variable at the boundary.

## Overview

This boundary condition enforces a specified gradient at the boundaries. It calculates the required macroscopic value at the boundary to satisfy the gradient condition and then uses the Dirichlet implementation logic to apply it. Note that the gradient specified at the boundary is not the gradient of macroscopic variable. Rather, it is the gradient of distribution fucntions (microscopci variable). This boundary conditions only computes missing directions by applying Non-Equilibrium Boundary Condition (NEBC). This means that the non-equilibirum part of the distribution function is computed from existing values in the current cell and the equilibrium part is computed from the prescribed gradients at the boundary.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` and `regional` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. The `regional` boundary enables users to mark different walls of the domain with different 
values and apply Neumann condition only in those regions.

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
