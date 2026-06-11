# NEML2GradientVector

!syntax description /TensorComputes/Solve/NEML2GradientVector

NEML2GradientVector computes the spatial gradient of a scalar buffer and returns a vector tensor (`neml2::Vec`) with components $(\partial_x u, \partial_y u, \partial_z u)$. When [!param](/TensorComputes/Solve/NEML2GradientVector/input_is_reciprocal) `= true`, the input is treated as already in reciprocal space.

## Overview

The gradient is computed spectrally using FFTs as

\begin{equation}
\nabla u = \mathcal{F}^{-1}\{ i \, \mathbf{k} \, \mathcal{F}\{u\}\}.
\end{equation}

In 2D/1D, the unused components are zero.

Requires NEML2 support to enable the `neml2::Vec` value type.

## Example Input File Syntax

!listing test/tests/typed_tensors/gradient.i block=TensorComputes/Initialize/grad_c

!syntax parameters /TensorComputes/Solve/NEML2GradientVector

!syntax inputs /TensorComputes/Solve/NEML2GradientVector

!syntax children /TensorComputes/Solve/NEML2GradientVector
