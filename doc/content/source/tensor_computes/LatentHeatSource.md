# LatentHeatSource

!syntax description /TensorComputes/Solve/LatentHeatSource

Computes the latent heat source term

\begin{equation}
L\frac{s - s^{n-1}}{\Delta t}
\end{equation}

using the value of the solid fraction buffer `s` from the previous solver substep,
where $\Delta t$ is the substep size (buffer states advance once per substep).

## Example Input File Syntax

!listing examples/thermal_multigrain_dendrites.i block=TensorComputes/Solve/latent

!syntax parameters /TensorComputes/Solve/LatentHeatSource

!syntax inputs /TensorComputes/Solve/LatentHeatSource

!syntax children /TensorComputes/Solve/LatentHeatSource
