# LBMPhaseEquilibrium

!syntax description /TensorComputes/Solve/LBMPhaseEquilibrium

This object computes equilibrium distribution for phase-field LBM.

## Overview

Builds equilibrium distributions for a phase field according to the chosen stencil and supplied
macroscopic fields. Provide the scalar phase field via
[!param](/TensorComputes/Solve/LBMPhaseEquilibrium/phi) and velocity via
[!param](/TensorComputes/Solve/LBMPhaseEquilibrium/velocity)

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Initialize/h_eq_init

!syntax parameters /TensorComputes/Solve/LBMPhaseEquilibrium

!syntax inputs /TensorComputes/Solve/LBMPhaseEquilibrium

!syntax children /TensorComputes/Solve/LBMPhaseEquilibrium
