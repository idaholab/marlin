# LBMAllenCahnSource

!syntax description /TensorComputes/Solve/LBMAllenCahnSource

This compute object adds the Allen-Cahn source term to the phase field distribution function for lattice Boltzmann simulations.

## Overview

Computes the conservative Allen\-Cahn source that couples the phase field order parameter to the
hydrodynamic velocity. The source includes an anti\-diffusion term that maintains the interface
thickness at $D$ lattice units. Provide the phase field via
[!param](/TensorComputes/Solve/LBMAllenCahnSource/phi), the velocity via
[!param](/TensorComputes/Solve/LBMAllenCahnSource/velocity), and the gradient via
[!param](/TensorComputes/Solve/LBMAllenCahnSource/grad_phi). The relaxation parameter is set with
[!param](/TensorComputes/Solve/LBMAllenCahnSource/tau) and interface thickness with
[!param](/TensorComputes/Solve/LBMAllenCahnSource/thickness).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/apply_forces_phase

!syntax parameters /TensorComputes/Solve/LBMAllenCahnSource

!syntax inputs /TensorComputes/Solve/LBMAllenCahnSource

!syntax children /TensorComputes/Solve/LBMAllenCahnSource
