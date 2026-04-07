# RandomRotationTensor

!syntax description /TensorComputes/Solve/RandomRotationTensor

Generates a constant 3x3 random rotation matrix (uniform over SO(3)) and
fills the output tensor with that value. The output buffer must use
`value_dimensions = '3 3'`.

## Example Input File Syntax

!listing examples/thermal_multigrain_dendrites.i block=TensorComputes/Initialize/rot1

!syntax parameters /TensorComputes/Solve/RandomRotationTensor

!syntax inputs /TensorComputes/Solve/RandomRotationTensor

!syntax children /TensorComputes/Solve/RandomRotationTensor
