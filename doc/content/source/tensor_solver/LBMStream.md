# LBMStream

!syntax description /TensorSolver/LBMStream

This tensor solver moves LBM distributions in time by streaming them based to the chosen stencil. The old state buffers must be post collision distributions. Multiple input and output buffers can be provided.

## Overview

Streams distribution functions along stencil directions to advance in time. Provide the active
distribution buffer via [!param](/TensorSolver/LBMStream/buffer) and the post\-collision history via
[!param](/TensorSolver/LBMStream/f_old).

The streaming step is governed by the equation:

$$
f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)
$$

where:
- $f_i$ is the distribution function for direction $i$.
- $f_i^*$ is the post-collision distribution function.
- $\mathbf{x}$ is the spatial position.
- $\mathbf{e}_i$ is the lattice velocity vector.
- $\Delta t$ is the time step.

In the implementation, this is achieved using `torch::roll`. For each discrete velocity direction $i$, the distribution tensor is shifted by the integer components of $\mathbf{e}_i$ (i.e., `_ex[i]`, `_ey[i]`, `_ez[i]`). This efficiently moves the population densities to their neighboring nodes in a single operation.

## Example Input File Syntax

!listing
[TensorSolver]
  type = LBMStream
  buffer = 'f g'
  f_old = 'fpc gpc'
[]
!listing-end


!syntax parameters /TensorSolver/LBMStream

!syntax inputs /TensorSolver/LBMStream

!syntax children /TensorSolver/LBMStream
