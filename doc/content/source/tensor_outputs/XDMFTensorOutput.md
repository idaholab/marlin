# XDMFTensorOutput

!syntax description /TensorOutputs/XDMFTensorOutput

This TensorOutput object writes tensors to an [XDMF](https://www.xdmf.org/) data file set.
The parameter [!param](/TensorOutputs/XDMFTensorOutput/enable_hdf5) enables writing of the raw
tensor data to an HDF5 file. The resulting data items will be bamed `/var.n`, where `var` is the name of the
exported tensor and `n` is the simulation step. Output as cell or node centered data is supported
and can be selected using the [!param](/TensorOutputs/XDMFTensorOutput/output_mode) parameter. Cell centered output
results in a value per simulation grid cell (e.g. `N[0] * N[1] * N[2]` entries), while for node centered output
the cell edge nodes are periodically replicated, resulting in `(N[0]+1) * (N[1]+1) * (N[2]+1)` exported entries.

!alert note title=Parallel limitation
When running with more than one MPI rank, this output currently supports only `CELL` mode. Each
rank writes its local data to rank-suffixed files and rank 0 emits the global XDMF collection that
references those blocks.

## Overview

Writes one or more buffers selected by
[!param](/TensorOutputs/XDMFTensorOutput/buffer) to an XDMF/HDF5 pair suitable for Paraview or
Visit. Enable HDF5 with [!param](/TensorOutputs/XDMFTensorOutput/enable_hdf5) and select cell or
node data with [!param](/TensorOutputs/XDMFTensorOutput/output_mode).

## Example Input File Syntax

!listing examples/lbm/Karman-vortex/cylinder.i block=TensorOutputs/xdmf2

!syntax parameters /TensorOutputs/XDMFTensorOutput

!syntax inputs /TensorOutputs/XDMFTensorOutput

!syntax children /TensorOutputs/XDMFTensorOutput
