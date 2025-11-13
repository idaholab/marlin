# TensorOutputs System

The `TensorOutputs` system is a custom asynchronous Marlin output system that directly writes tensor data to files without
projection to a mesh. At the time of outputting it creates copies ofa all tensors slated for output and launches the
respective `TensorOutputs` in a separate thread. This allows the simulation to progress while data is written to disk, resulting
in a high compute device utilization.

## Example Input File Syntax

!! Describe and include an example of how to use the TensorOutputs system.

!syntax list /TensorOutputs objects=True actions=False subsystems=False

!syntax list /TensorOutputs objects=False actions=False subsystems=True

!syntax list /TensorOutputs objects=False actions=True subsystems=False
