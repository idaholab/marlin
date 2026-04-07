# LatentHeatSource

!syntax description /TensorComputes/Solve/LatentHeatSource

Computes the latent heat source term

\begin{equation}
L\frac{s - s^{n-1}}{\Delta t}
\end{equation}

using the previous value of the solid fraction buffer `s`.

## Example Input File Syntax

!listing examples/thermal_multigrain_dendrites.i block=TensorComputes/Solve/latent

!syntax parameters /TensorComputes/Solve/LatentHeatSource

!syntax inputs /TensorComputes/Solve/LatentHeatSource

!syntax children /TensorComputes/Solve/LatentHeatSource
