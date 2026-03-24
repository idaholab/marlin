# LBMPressureCorrectedEquilibrium

!syntax description /TensorComputes/Solve/LBMPressureCorrectedEquilibrium

This compute object evaluates the pressure-corrected equilibrium distribution function for the phase field lattice Boltzmann model.

## Overview

Computes the equilibrium distribution $g_i^{\text{eq}}$ using a pressure\-based formulation where
the zeroth moment recovers pressure rather than density. For $i = 0$,
$g_0^{\text{eq}} = \frac{p}{c_s^2}(w_0 - 1) + \rho\,s_0(\mathbf{u})$; for $i \neq 0$,
$g_i^{\text{eq}} = \frac{p}{c_s^2}\,w_i + \rho\,s_i(\mathbf{u})$. Provide the density via
[!param](/TensorComputes/Solve/LBMPressureCorrectedEquilibrium/rho), velocity via
[!param](/TensorComputes/Solve/LBMPressureCorrectedEquilibrium/velocity), and pressure via
[!param](/TensorComputes/Solve/LBMPressureCorrectedEquilibrium/pressure).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/f_eq

!syntax parameters /TensorComputes/Solve/LBMPressureCorrectedEquilibrium

!syntax inputs /TensorComputes/Solve/LBMPressureCorrectedEquilibrium

!syntax children /TensorComputes/Solve/LBMPressureCorrectedEquilibrium
