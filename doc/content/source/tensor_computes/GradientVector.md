# GradientVector

!syntax description /TensorComputes/Solve/GradientVector

GradientVector computes the spatial gradient of a scalar buffer and returns a torch tensor with components $(\partial_x u, \partial_y u, \partial_z u)$ stacked along the trailing dimension. When [!param](/TensorComputes/Solve/GradientVector/input_is_reciprocal) `= true`, the input is treated as already in reciprocal space.

## Overview

The gradient is computed spectrally using FFTs as

\begin{equation}
\nabla u = \mathcal{F}^{-1}\{ i \, \mathbf{k} \, \mathcal{F}\{u\}\}.
\end{equation}

In 2D/1D, the unused components are zero.

## Example Input File Syntax

!listing test/tests/typed_tensors/gradient_vector.i block=TensorComputes/Initialize/grad_c

!syntax parameters /TensorComputes/Solve/GradientVector

!syntax inputs /TensorComputes/Solve/GradientVector

!syntax children /TensorComputes/Solve/GradientVector
