# Real-Space Parallelization

This document describes the real-space domain decomposition path, its data layout, and the APIs/expectations for computes and outputs. It follows the style of `doc/content/jit_tracing.md`.

## Overview

`parallel_mode = REAL_SPACE` partitions the physical domain into near-cubic subdomains and uses point-to-point halo exchanges for ghost layers. No FFTs are available in this mode; reciprocal axes/shapes are left empty. All tensors on a rank share the same local shape: **owned cells + symmetric ghost layers (Gmax) in every spatial direction**.

## Domain Partitioning

- Ranks are factored into `Nx × Ny × Nz` (Nz=1 in 2D) to minimize surface area/aspect imbalance.
- Each rank stores begin/end indices per dimension; periodic directions are configured via `[Domain] periodic_directions = 'X Y Z'`.
- For periodic single-partition axes, wrap-around halos are self-copied. For partitioned periodic axes, halos wrap to opposite ranks.
- FFT/ifft calls throw in REAL_SPACE; reciprocal tensors are disallowed.

## Ghost Layers and Tensor Sizing

- Computes request ghost width per input via `getInputBuffer(..., ghost_layers)`. TensorProblem tracks the maximum (`Gmax`) and sizes all buffers to owned+2*Gmax.
- ParsedCompute pads coordinate tensors (x,y,z) to Gmax in REAL_SPACE so expressions using extra symbols yield padded outputs. Other inputs that are already padded are left as-is; padding is only added if spatial dims are smaller than the target local tensor shape.
- TensorBufferBase exposes `ownedView()` to slice off ghosts on the leading spatial dims.

## Communication

- TensorProblem performs halo exchanges before each compute/IC/PP for inputs requesting ghosts.
- Exchange pattern:
  - For partitioned axes, neighbors are lower/upper ranks; if periodic and at a boundary, wrap to the opposite side; if the wrap resolves to self, copy locally.
  - Non-blocking pattern: post all Irecv, post all Isend, waitall, then copy into ghost views.
  - Tags are deterministic per dimension/side (base 200).
- GPU-aware MPI is used if enabled; otherwise, CPU staging buffers are used.

## Computes

- FiniteDifferenceLaplacian supports `stencil_width = 3|5` (default 3). Padding equals the radius; convolution output matches input size and is copied directly into the output buffer.
- Computes should assume inputs include ghosts they requested and may write only owned regions if needed; outputs are sized to the padded shape.

## Outputs and Postprocessing

- XDMFTensorOutput uses `ownedView()` to drop ghosts before transposing/writing, so ghost layers are never emitted.
- TensorExtremeValuePostprocessor (and other PP using TensorPostprocessor base) now access the buffer via its base and operate on `ownedView()`, excluding ghosts.

## Configuration Notes

- REAL_SPACE forbids reciprocal buffers and FFT usage.
- Periodic directions must be declared for correct wrap halos; otherwise, boundary halos remain zero.
- Set `[Domain] debug = true` to print halo exchange details (rank, neighbor, offsets, counts).

## Limitations / Future

- Boundary conditions beyond periodic/implicit zero (Dirichlet/Neumann objects) are TBD.
- Additional buffer accessors for TensorBufferBase (views, ownership metadata) may be added as needed for advanced postprocessing.*** End Patch
