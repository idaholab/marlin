# LBMComputePhysicalVelocity

!syntax description /TensorComputes/Solve/LBMComputePhysicalVelocity

## Overview

The purpose of this object to convert the dimensionless LBM velocity into its physical counterpart. This is done using the following relationships:

!equation
\vec{u} = \vec{u}^{*}\frac{\Delta x}{\Delta t},

!equation
\nu = C_s^2\left(\tau^{*} - 0.5\right) \frac{\Delta x^2}{\Delta t},

where,

- $\vec{u}$ is the physical velocity ([!param](/TensorComputes/Solve/LBMComputePhysicalVelocity/buffer)).
- $u^{*}$ is the LBM velocity ([!param](/TensorComputes/Solve/LBMComputePhysicalVelocity/velocity)).
- $\Delta x$ is the grid size, infered from the dimensions specified in the [Domain](DomainAction.md) bloc.
- $\Delta t$ is the timestep size, computed via the second equation
- $\nu$ is the physical kinematic viscosity ([!param](/TensorComputes/Solve/LBMComputePhysicalVelocity/nu)).
- $\tau^{*}$ is the relaxation parameter ([!param](/TensorComputes/Solve/LBMComputePhysicalVelocity/tau)).
- $C_s$ is the lattice speed of sound ($\frac{1}{\sqrt{3}}$).

## Example Input Syntax

The following is a stripped-down example of adding physical velocity postprocessing. The collision compute and problem are shown to illustrate the relationship of `tau`.

!listing lbm/convert_to_physical_2D.i
    block=TensorBuffers/physical_velocity
          TensorComputes/Solve/collision
          TensorComputes/Postprocess/physical_velocity
          Problem

!syntax parameters /TensorComputes/Solve/LBMComputePhysicalVelocity

!syntax inputs /TensorComputes/Solve/LBMComputePhysicalVelocity

!syntax children /TensorComputes/Solve/LBMComputePhysicalVelocity

