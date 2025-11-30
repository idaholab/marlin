# AddLBMBCAction

!syntax description /TensorComputes/Boundary/AddLBMBCAction

The AddLBMBCAction adds boundary conditions for Lattice Boltzmann simulations.

## Overview

This action is responsible for instantiating and registering Lattice Boltzmann boundary condition objects defined in the input file. It operates within the `[TensorComputes]` block under the `[Boundary]` sub-block.

When a user defines a boundary condition (such as `LBMDirichletBC`, `LBMNeumannBC`, or `LBMFixedFirstOrderBC`), this action ensures it is added to the `LatticeBoltzmannProblem`. It verifies that the current problem type is indeed a `LatticeBoltzmannProblem` before proceeding.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorComputes/Boundary

!syntax parameters /TensorComputes/Boundary/AddLBMBCAction
