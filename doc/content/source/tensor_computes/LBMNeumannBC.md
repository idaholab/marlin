# LBMNeumannBC

!syntax description /TensorComputes/Boundary/LBMNeumannBC

Implements a Neumann boundary condition for Lattice Boltzmann simulations. It enforces a fixed gradient of a macroscopic variable at the boundary.

## Overview

This boundary condition enforces a specified gradient at the boundaries. It calculates the required macroscopic value at the boundary to satisfy the gradient condition and then uses the Dirichlet implementation logic to apply it.

It supports both domain boundaries (top/bottom) and internal boundaries defined by a region ID.

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
