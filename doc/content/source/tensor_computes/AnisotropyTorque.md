# AnisotropyTorque

!syntax description /TensorComputes/Solve/AnisotropyTorque

AnisotropyTorque computes the anisotropy torque by summing FFT gradient components of a vector-valued `dmu_dn` field multiplied by a scalar weighting field `g`.

## Overview

The object expects:

- `dmu_dn`: a vector-valued tensor buffer with trailing component dimension `3`
- `g`: a scalar tensor buffer with the same spatial shape

It computes

\begin{equation}
\sum_{i=0}^{\mathrm{dim}-1} \partial_i \left(d\mu/dn_i \cdot g\right)
\end{equation}

using the same FFT-based gradient machinery as the other gradient computes.

## Example Input File Syntax

!listing test/tests/anisotropic_grain_growth/circular_grain.i block=TensorComputes/Solve/anisotropy_torque

!syntax parameters /TensorComputes/Solve/AnisotropyTorque

!syntax inputs /TensorComputes/Solve/AnisotropyTorque

!syntax children /TensorComputes/Solve/AnisotropyTorque
