# LBMEquilibrium

!syntax description /TensorComputes/Solve/LBMEquilibrium

Compute LBM equilibrium distribution functions. The equilibrium uses a second-order Hermite
expansion of the Maxwell-Boltzmann distribution.

## Overview

Given bulk scalar `bulk` (e.g., density or temperature) field and velocity vector field `velocity` field, this
object computes `f_eq` on all lattice directions using the stencil weights and the lattice sound
speed $c_s$ defined by the `LatticeBoltzmannProblem`.

The equilibrium distribution function $f_i^{eq}$ is computed as:

$$
f_i^{eq} = w_i \rho \left( 1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{u^2}{2 c_s^2} \right)
$$

where:
- $w_i$ are the lattice weights
- $\rho$ is the macroscopic density (or other scalar quantity)
- $\mathbf{e}_i$ are the discrete lattice velocities
- $\mathbf{u}$ is the macroscopic velocity
- $c_s$ is the lattice speed of sound

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [feq]
      type = LBMEquilibrium
      buffer = feq
      bulk = rho
      velocity = u
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMEquilibrium

!syntax inputs /TensorComputes/Solve/LBMEquilibrium

!syntax children /TensorComputes/Solve/LBMEquilibrium
