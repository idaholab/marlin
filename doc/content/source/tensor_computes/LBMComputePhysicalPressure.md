# LBMComputePhysicalPressure

!syntax description /TensorComputes/Solve/LBMComputePhysicalPressure

## Overview

The purpose of this object to convert the dimensionless LBM density into a physical gauge pressure. This is done using the following relationships:

!equation
P = \rho_0 C_s^2 \left(\frac{\Delta x}{\Delta t}\right)^2 \rho^{*},

!equation
\nu = C_s^2\left(\tau^{*} - 0.5\right) \frac{\Delta x^2}{\Delta t},

!equation
P_{\mathrm{gauge}} = P - P_{\mathrm{ref}} = \rho_0 C_s^2 \left(\frac{\Delta x}{\Delta t}\right)^2 \left(\rho^{*} - \rho^{*}_{\mathrm{ref}}\right),

where,

- $P_{\mathrm{gauge}}$ is the gauge pressure ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/buffer))
- $\rho^{*}$ is the LBM density ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/rho)).
- $\rho_0$ is the density conversion factor, usually the reference density of the physical fluid ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/rho0_phys)).
- $\rho^{*}_{\mathrm{ref}}$ is the reference LBM density, typically what is set when initializing the density tensor buffer ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/rho0)). Setting this to 0 will give an absolute pressure.
- $C_s$ is the lattice speed of sound ($\frac{1}{\sqrt{3}}$).
- $\Delta x$ is the grid size, infered from the dimensions specified in the [Domain](DomainAction.md) bloc.
- $\Delta t$ is the timestep size, computed via the second equation
- $\nu$ is the physical kinematic viscosity ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/nu)).
- $\tau^{*}$ is the relaxation parameter ([!param](/TensorComputes/Solve/LBMComputePhysicalPressure/tau)).


## Example Input Syntax

The following is a stripped-down example of adding physical pressure postprocessing. The collision compute and problem are shown to illustrate the relationship of `tau`. The density initialization is shown to illustrate the relationship of [!param](/TensorComputes/Solve/LBMComputePhysicalPressure/rho0) and the LBM density initialization.

!listing lbm/convert_to_physical_2D.i
    block=TensorBuffers/physical_pressure
          TensorComputes/Initialize/initial_density
          TensorComputes/Solve/collision
          TensorComputes/Postprocess/physical_pressure
          Problem

!syntax parameters /TensorComputes/Solve/LBMComputePhysicalPressure

!syntax inputs /TensorComputes/Solve/LBMComputePhysicalPressure

!syntax children /TensorComputes/Solve/LBMComputePhysicalPressure

