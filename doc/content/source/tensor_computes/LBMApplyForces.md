# LBMApplyForces

!syntax description /TensorComputes/Solve/LBMApplyForces

LBMApplyForces adds forces onto LBM distribution fucntion. The forces act as source term.

## Overview

Applies a body force term to a post\-collision distribution using the Guo forcing scheme. Provide
the veloicty via [!param](/TensorComputes/Solve/LBMApplyForces/velocity), density via 
[!param](/TensorComputes/Solve/LBMApplyForces/rho) and the force vector via
[!param](/TensorComputes/Solve/LBMApplyForces/forces). The relaxation time is referenced via
[!param](/TensorComputes/Solve/LBMApplyForces/tau0).

## Example Input File Syntax

!listing examples/lbm/natural_convection_3D/convection.i block=TensorComputes/Solve/Apply_forces

!syntax parameters /TensorComputes/Solve/LBMApplyForces

!syntax inputs /TensorComputes/Solve/LBMApplyForces

!syntax children /TensorComputes/Solve/LBMApplyForces
