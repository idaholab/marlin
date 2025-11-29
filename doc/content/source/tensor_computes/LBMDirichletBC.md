# LBMDirichletBC

!syntax description /TensorComputes/Boundary/LBMDirichletBC

Implements a Dirichlet boundary condition for Lattice Boltzmann simulations. It fixes the value of a macroscopic scalar variable (like density or temperature) at the boundary.

## Overview

This boundary condition enforces a specified value at the boundaries. It computes all directions by applying Non-Equilibrium Extrapolation Method (NEEM). This means that the non-equilibirum part of the distribution function is computed from the existing values in the current cell and the equilibrium part is computed from the prescribed values at the boundary.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` and `regional` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. The `regional` boundary enables users to mark different walls of the domain with different 
values and apply Dirichlet/Neumann condition only in those regions.

The boundary condition is applied using the Non-Equilibrium Extrapolation Method (NEEM). The distribution function at the boundary node $\mathbf{x}_b$ is updated as:

$$
f_i(\mathbf{x}_b, t) = f_i^{eq}(\rho_b, \mathbf{u}_b) + (f_i(\mathbf{x}_b, t) - f_i^{eq}(\rho(\mathbf{x}_b), \mathbf{u}(\mathbf{x}_b)))
$$

where:
- $f_i^{eq}(\rho_b, \mathbf{u}_b)$ is the equilibrium distribution computed using the prescribed boundary values (e.g., fixed density $\rho_b$ or temperature).
- $f_i(\mathbf{x}_b, t) - f_i^{eq}(\rho(\mathbf{x}_b), \mathbf{u}(\mathbf{x}_b))$ represents the non-equilibrium part, which is extrapolated from the current state of the fluid at the boundary node.

This method assumes that the non-equilibrium part of the distribution function is continuous across the boundary.

Please note that, this boundary condition is not recommended for prescribed velocity boundary conditions. For that, please use `LBMFixedFirstOrderBC`

## Example Input File Syntax

!listing
[TensorComputes]
  [Boundary]
    [regional]
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
