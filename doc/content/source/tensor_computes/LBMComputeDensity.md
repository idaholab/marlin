# LBMComputeDensity

!syntax description /TensorComputes/Solve/LBMComputeDensity

This compute object sums LBM distribution functions along the last dimension to compute density or temperature or any other macroscopic scalar buffer depending on the physics of the problem.

## Overview

Given a distribution tensor $f(x,y,z, q)$, computes the macroscopic scalar quantity
$rho(x,y,z) = \Sigma_q f$. Values on solid cells are set to zero using the lattice mask.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [rho]
      type = LBMComputeDensity
      buffer = rho
      f = f
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMComputeDensity

!syntax inputs /TensorComputes/Solve/LBMComputeDensity

!syntax children /TensorComputes/Solve/LBMComputeDensity
