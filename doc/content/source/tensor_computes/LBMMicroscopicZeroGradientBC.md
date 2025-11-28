# LBMMicroscopicZeroGradientBC

!syntax description /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

This object implements zero flux Neumann BC on LBM distribution functions.

## Overview

Imposes zero normal gradient on the microscopic distributions at selected domain faces. Choose
faces via [!param](/TensorComputes/Boundary/LBMMicroscopicZeroGradientBC/boundary) and provide the
target distribution via [!param](/TensorComputes/Boundary/LBMMicroscopicZeroGradientBC/buffer).

## Example Input File Syntax

!listing test/tests/lbm/obstacle.i block=TensorComputes/Boundary/right

!syntax parameters /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

!syntax inputs /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

!syntax children /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC
