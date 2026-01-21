# AddLBMStencilAction
!syntax description /Stencil/AddLBMStencilAction

The AddLBMStencilAction adds stencils, also known as decriptors, for Lattice Botlzmann simulations.
For 2D simulations, D2Q9 stencil is available `(/LBMD2Q9.md)`. For 3D simulations the available stencils are D3Q19 `(/LBMD3Q19.md)` and D3Q27 `(/LBMD3Q27.md)`

## Overview

This action is responsible for creating and registering the Lattice Boltzmann stencil (or descriptor) used in the simulation. The stencil defines the discrete velocity set, weights, relaxation matrices for MRT collision, transformation matrices and other lattice properties required for the LBM algorithm.

It operates within the `[Stencil]` block in the input file. When a stencil type (like `LBMD2Q9`, `LBMD3Q19`, or `LBMD3Q27`) is specified, this action adds it to the `LatticeBoltzmannProblem`.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=Stencil

!syntax parameters /Stencil/AddLBMStencilAction
