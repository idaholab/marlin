# LBMDirichletBC

!syntax description /TensorComputes/Boundary/LBMDirichletBC

Implements a Dirichlet boundary condition for Lattice Boltzmann simulations. It fixes the value of a macroscopic variable (like density or temperature) at the boundary.

## Overview

This boundary condition enforces a fixed value at the specified boundaries. It computes the equilibrium distribution function based on the prescribed boundary value and the local velocity, and then applies a non-equilibrium extrapolation scheme to determine the incoming distribution functions.

It supports both domain boundaries (top, bottom, etc.) and internal boundaries defined by a region ID in binary media.

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
