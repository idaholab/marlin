# LBMDirichletWallBC

!syntax description /TensorComputes/Solve/LBMDirichletWallBC

This LBMDirichletWallBC objects implements preliminary version of Dirichlet boundary condition at the solid walls.

## Overview

Imposes a fixed value on a macroscopic field at solid walls by reconstructing incoming
distributions to enforce the desired boundary condition. Select the target field via
[!param](/TensorComputes/Solve/LBMDirichletWallBC/buffer) and choose faces using
[!param](/TensorComputes/Solve/LBMDirichletWallBC/boundary).

## Requirements

**This boundary condition requires `binary_media` to be defined in LatticeBoltzmannProblem.**
It automatically identifies fluid cells adjacent to solid regions (where binary_media = 0) and
applies the Dirichlet condition at the fluid-solid interface.

Always use `boundary = wall` when specifying this boundary condition, as it operates on
internal solid boundaries defined by the binary media tensor, not on domain edges.

## Example Input File Syntax

!listing test/tests/lbm/dirichlet_box.i block=TensorComputes/Boundary/

!syntax parameters /TensorComputes/Solve/LBMDirichletWallBC

!syntax inputs /TensorComputes/Solve/LBMDirichletWallBC

!syntax children /TensorComputes/Solve/LBMDirichletWallBC
