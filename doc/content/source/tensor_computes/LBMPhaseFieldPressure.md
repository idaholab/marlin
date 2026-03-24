# LBMPhaseFieldPressure

!syntax description /TensorComputes/Solve/LBMPhaseFieldPressure

This compute object computes the pressure field for the phase field lattice Boltzmann model.

## Overview

Evaluates the macroscopic pressure from the pressure\-corrected distribution function using the
relation $p = \frac{c_s^2}{1 - w_0}\left[\sum_{i \neq 0} g_i + \frac{\Delta t}{2}(\rho_l - \rho_g)\,\mathbf{u}\cdot\nabla\phi + \rho\,s_0(\mathbf{u})\right]$.
Provide the distribution function via
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/f), velocity via
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/velocity), gradient via
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/grad_phi), and density via
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/rho). Liquid and gas densities are set with
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/rho_l) and
[!param](/TensorComputes/Solve/LBMPhaseFieldPressure/rho_g).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/pressure

!syntax parameters /TensorComputes/Solve/LBMPhaseFieldPressure

!syntax inputs /TensorComputes/Solve/LBMPhaseFieldPressure

!syntax children /TensorComputes/Solve/LBMPhaseFieldPressure
