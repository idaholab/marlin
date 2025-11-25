# LBMDirichletBC

!syntax description /TensorComputes/Boundary/LBMDirichletBC

Implements a Dirichlet boundary condition for Lattice Boltzmann simulations. It fixes the value of a macroscopic variable (like density or temperature) at the boundary.

## Overview

This boundary condition enforces a specified value at the boundaries. This boundary conditions computes all directions by applying Non-Equilibrium Boundary Condition (NEBC). This means that the non-equilibirum part of the distribution function is computed from the existing values in the current cell and the equilibrium part is computed from the prescribed values at the boundary.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` and `regional` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. The `regional` boundary enables users to mark different walls of the domain with different 
values and apply Neumann condition only in those regions.

Please note that, this boundary condition is not recommended for prescribed velocity boundary conditions. For that, please use `LBMFixedFirstOrderBC`

## Example Input File Syntax

!listing
[TensorComputes]
  [Boundary]
    [walls]
      type = LBMDirichletBC
      buffer = g
      f_old = gpc
      feq=geq
      velocity = velocity
      rho = T
      value = 1.0
      region_id = 2
      boundary = regional
    []
  []
[]

!listing-end

!syntax parameters /TensorComputes/Boundary/LBMDirichletBC

!syntax inputs /TensorComputes/Boundary/LBMDirichletBC

!syntax children /TensorComputes/Boundary/LBMDirichletBC
