# LBMComputeResidual

!syntax description /TensorComputes/Solve/LBMComputeResidual

This object simply computes the L1 difference between current and old timestep scalar vectors. This difference is used as a residual in LBM simulations to tell how far the simulaion is from steady-state.

## Overview

Computes a simple residual measure from a supplied buffer, e.g., for convergence monitoring. Use
[!param](/TensorComputes/Solve/LBMComputeResidual/buffer) to select the destination and
[!param](/TensorComputes/Solve/LBMComputeResidual/speed) or another input for the source.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorComputes/Solve/residual

!syntax parameters /TensorComputes/Solve/LBMComputeResidual

!syntax inputs /TensorComputes/Solve/LBMComputeResidual

!syntax children /TensorComputes/Solve/LBMComputeResidual
