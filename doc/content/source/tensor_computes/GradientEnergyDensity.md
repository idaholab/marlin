# GradientEnergyDensity

!syntax description /TensorComputes/Solve/GradientEnergyDensity

GradientEnergyDensity computes the gradient energy density of a set of order parameters from
their gradient vector buffers (e.g. as produced by [GradientVector.md]).

## Overview

The object expects `gradient_buffers` to list one vector-valued tensor buffer per order
parameter, each with trailing component dimension `3` and matching spatial shape. It computes

\begin{equation}
\frac{1}{2} \kappa \sum_{i} |\nabla \eta_i|^2
\end{equation}

where the sum runs over the listed buffers and $\kappa$ is the `kappa` parameter. No FFT is
performed by this object; it operates directly on gradient buffers already computed elsewhere
(typically by [GradientVector.md]), so it adds no redundant transforms when those buffers already
exist.

## Example Input File Syntax

!listing test/tests/anisotropic_grain_growth/sigma3_circular_grain.i block=TensorComputes/Solve/gradient_energy

!syntax parameters /TensorComputes/Solve/GradientEnergyDensity

!syntax inputs /TensorComputes/Solve/GradientEnergyDensity

!syntax children /TensorComputes/Solve/GradientEnergyDensity
