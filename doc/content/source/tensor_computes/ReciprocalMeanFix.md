# ReciprocalMeanFix

!syntax description /TensorComputes/Solve/ReciprocalMeanFix

Enforces a target mean value by setting the k=0 Fourier mode of a
reciprocal-space tensor to `u_inf * Ncells`.

## Example Input File Syntax

!listing examples/thermal_multigrain_dendrites.i block=TensorComputes/Solve/ubar_fixed

!syntax parameters /TensorComputes/Solve/ReciprocalMeanFix

!syntax inputs /TensorComputes/Solve/ReciprocalMeanFix

!syntax children /TensorComputes/Solve/ReciprocalMeanFix
