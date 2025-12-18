# LBMMicroscopicZeroGradientBC

!syntax description /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

This object implements zero flux Neumann BC on LBM distribution functions.

## Overview

Imposes a zero normal gradient condition on the microscopic distribution functions at selected domain faces. This is achieved by setting the distribution function at the boundary node equal to the value of its immediate interior neighbor (0th-order extrapolation).

This boundary condition is often used as a simple **outflow** condition, allowing structures to leave the domain with minimal reflection.

Supported boundaries are domain faces: `left`, `right`, `top`, `bottom`, `front`, and `back`. Internal `wall` boundaries are not supported.

Choose faces via [!param](/TensorComputes/Boundary/LBMMicroscopicZeroGradientBC/boundary) and provide the target distribution via [!param](/TensorComputes/Boundary/LBMMicroscopicZeroGradientBC/buffer).

## Example Input File Syntax

!listing test/tests/lbm/obstacle.i block=TensorComputes/Boundary/right

!syntax parameters /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

!syntax inputs /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC

!syntax children /TensorComputes/Boundary/LBMMicroscopicZeroGradientBC
