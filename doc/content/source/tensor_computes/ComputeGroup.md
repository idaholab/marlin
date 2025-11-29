# ComputeGroup

!syntax description /TensorComputes/Solve/ComputeGroup

Combines multiple compute operators into one execution unit and applies dependency resolution among
them. Compute groups may be nested. Inputs of a group are all inputs of the members that are not
produced by any member; outputs are all members' outputs not consumed by any member. Provide the
members with [!param](/TensorComputes/Solve/ComputeGroup/computes).

## JIT Tracing

ComputeGroup supports JIT (Just-In-Time) compilation of tensor operations using PyTorch's tracing
infrastructure. When enabled (the default), sequences of compatible tensor operations are captured
into optimized computation graphs that can reduce memory usage and improve performance.

JIT tracing is controlled by the [!param](/TensorComputes/Solve/ComputeGroup/enable_jit) parameter.
Operations that cannot be JIT traced (such as random number generation or iterative solvers) are
automatically detected and the execution is split into traced and non-traced segments.

For detailed information about JIT tracing, see [JIT Tracing System](jit_tracing.md).

## Example Input File Syntax

!listing test/tests/tensor_compute/group.i block=TensorComputes/Solve/group

!syntax parameters /TensorComputes/Solve/ComputeGroup

!syntax inputs /TensorComputes/Solve/ComputeGroup

!syntax children /TensorComputes/Solve/ComputeGroup
