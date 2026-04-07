# StackTensors

!syntax description /TensorComputes/Solve/StackTensors

Stacks a list of scalar tensor buffers into a single vector-valued tensor using
[!param](/TensorComputes/Solve/StackTensors/stack_dim) as the new dimension.

## Example Input File Syntax

```
[TensorComputes]
  [Solve]
    [gradvec]
      type = StackTensors
      buffer = gradvec
      inputs = 'gradx grady gradz'
    []
  []
[]
```

!syntax parameters /TensorComputes/Solve/StackTensors

!syntax inputs /TensorComputes/Solve/StackTensors

!syntax children /TensorComputes/Solve/StackTensors
