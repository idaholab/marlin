# Marlin Application Syntax

Marlin uses the MOOSE hierarchial input text (HIT) format, but adds a whole bunch
of new toplevel syntax. In addition some syntax allows for hierarchial nesting
with a semantic meaning. In the `[TensorCompute]` syntax the nesting is used to
group computes.

## `[Domain]`

The domain block sets up the problem domain. As Marlin is a grid based FFT code
it does not use a libMesh mesh (`[Mesh]` block) as a computation domain. However
a mesh can be set up to automatically match the grid domain if coupling to a
finite element run is desired. In that case Marlin supports fast copying of grid
based tensor buffers to the mesh and back.

The [DomainAction](DomainAction.md) also automatically sets up a
[TensorProblem](TensorProblem.md) object if the user didn;t explicitly specify a
problem type. Note that to use any Marlin objects the problem type has to be
TensorProblem or a class derived from TensorProblem.

## `[TensorComputes]`

TensorComputes (or tensor operators) are the Marlin explicit equivalent to
Kernels in MOOSE. Instead of computing point forces (and Jacobians) tensor
computes perform an explicit operation on a set of input buffers (tensors)
resulting in a set of (one or more) output buffers. Marlin performs automatic
dependency resolution to sort the tensor computes according to requested inputs
and outputs.

## `[TensorSolver]`

The TensorSolver advances the problem to the next time step (or rather sub step)
and closes the dependency cycle.

## `[TensorOutputs]`

While output using meshbased MOOSE outputs, such as Exodus, is possible, Marlin
provides a faster system to output tensor grids directly using a threaded output
system. Especially when running simulations on GPU devices this asynchronous
output is running on a CPU copy of the output buffers, while the next timestep
is already computing on the GPU. This allows for GPU utilizations up to 100%.

# Syntax

!syntax complete groups=MarlinApp level=1
