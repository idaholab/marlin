# DomainAction

!syntax description /Domain/DomainAction

## Overview

`DomainAction` configures the FFT simulation domain and compute devices, and wires up
the minimal runtime needed for Marlin problems:

- Sets problem dimension and grid resolution ([!param](/Domain/dim), [!param](/Domain/nx), [!param](/Domain/ny), [!param](/Domain/nz)) and computes grid spacing from [!param](/Domain/xmin)/[!param](/Domain/xmax), [!param](/Domain/ymin)/[!param](/Domain/ymax), [!param](/Domain/zmin)/[!param](/Domain/zmax).
- Builds real-space axes (cell-centered) and reciprocal-space axes (via `fftfreq/rfftfreq`, scaled by `2π`) and exposes helper accessors used throughout Marlin.
- Partitions the domain for parallel execution based on [!param](/Domain/parallel_mode) and optional [!param](/Domain/device_names)/[!param](/Domain/device_weights).
- Creates a matching mesh automatically (optional) using [!param](/Domain/mesh_mode) and adds a `DomainMeshGenerator`.
- Creates a `TensorProblem` by default if one is not provided, enabling all Marlin objects to run.

### Key concepts and behavior

- Dimension and grids: The domain is a structured grid. Grid spacing is `(max-min)/n` per axis; axes are built at cell centers. The reciprocal axes use Torch FFT frequency helpers and represent angular wavenumbers.
- Shapes and sizes: Methods such as `getShape()`, `getReciprocalShape()`, `getGridSize()`, `getReciprocalGridSize()` provide local/global extents used by tensor computes and outputs.
- On-demand tensors: `getXGrid()`, `getKGrid()`, and `getKSquare()` lazily build coordinate arrays for the local partition when first requested.
- FFT helpers: `fft()`/`ifft()` dispatch to serial or parallel implementations depending on `parallel_mode`. In serial, 1D/2D/3D transforms call `torch::fft::rfft{,2,n}` and inverse `irfft{,2,n}` with appropriate dimension lists. Slab/pencil modes target distributed FFTs (pencil not yet implemented).
- Reductions: `sum()` and `average()` reduce over the spatial dimensions. Note: reductions are implemented in serial; MPI reduction is not yet provided.
- Device and precision: If [!param](/Domain/device_names) are given, Marlin assigns a device per local host-rank and sets Torch floating precision with [!param](/Domain/floating_precision).

### Mesh generation and problem creation

- [!param](/Domain/mesh_mode) = `DOMAIN` generates a mesh with one element per grid cell; `DUMMY` creates a single-element mesh spanning the domain; `MANUAL` disables automatic mesh creation (user supplies `[Mesh]`). When mesh creation is enabled, `DomainAction` injects `SetupMeshAction` and adds a `DomainMeshGenerator` with consistent domain bounds and resolution.
- If no `Problem` is provided, `DomainAction` creates a `TensorProblem` during `create_problem_custom` to ensure Marlin components can run.

### Parallel partitioning modes

!alert note title=Work in progress
Parallel decomposition is a work in progress and not ready for production use!

- `NONE`: No decomposition; requires running in serial (`comm.size() == 1`).
- `FFT_SLAB`: Slab decomposition for 2D/3D. Real space is partitioned into X–Z slabs stacked along Y; in Fourier space Y–Z slabs stacked along X. Requires one all-to-all per FFT.
- `FFT_PENCIL`: Intended for 3D pencil decomposition with two many-to-many communications per FFT. Not implemented yet.


### Device assignment and weights

- [!param](/Domain/device_names): Names in Torch syntax (e.g., `"cuda:0"`, `"cuda:1"`, `"cpu"`). For MPI runs, each process determines its local host-rank and picks a device by index (modulo the list length).
- [!param](/Domain/device_weights): Optional relative speeds to influence partitioning balance across local ranks. If omitted, weights default proportionally to the number of devices.
- [!param](/Domain/floating_precision): Selects `DEVICE_DEFAULT`, `SINGLE`, or `DOUBLE` precision for Torch tensors on the chosen device.

## Example Input File Syntax

Basic 2D domain with a dummy mesh and serial execution

!listing test/tests/lbm/channel2D.i block=Domain

3D domain with a dummy mesh

!listing test/tests/lbm/channel3D.i block=Domain

Parallel slab decomposition with explicit device selection and weights

!listing test/tests/tensor_compute/parallel.i block=Domain

!syntax description /Domain/DomainAction

!syntax parameters /Domain/DomainAction

## Notes and Limitations

- [!param](/Domain/parallel_mode) = `NONE` requires a single-process run.
- `FFT_PENCIL` is declared but not implemented; prefer `FFT_SLAB` for distributed runs.
- `sum()`/`average()` use local reductions only; parallel reductions are not yet available.
- When [!param](/Domain/device_names) are omitted in MPI runs, the action aborts with an error to avoid ambiguous device assignment.

## Related

- `DomainMeshGenerator`: Mesh generator added automatically when `mesh_mode != MANUAL`.
- `TensorProblem`: Created by default if the user does not provide a `[Problem]` compatible with Marlin.
