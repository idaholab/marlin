# LatticeBoltzmannProblem

!syntax description /Problem/LatticeBoltzmannProblem

The problem object derived from TensorProblem (/TensorProblem.md) is used to solve Lattice Boltzmann simulations. It works in the same way as TensorProblem, with a few extra features. It adds stencil and boundary condition object and modifies the initialization and execution order in favor of LBM.

## Overview

Problem driver specialized for Lattice Boltzmann simulations: manages stencils, distribution buffers,
macroscopic fields, and substepping between collision and streaming. Provides access to LBM
constants like `c_s` and time step for objects that require them.

## Binary Media for Complex Geometries

The `binary_media` parameter enables simulation of flow through complex solid geometries (porous media, obstacles, etc.). It accepts an integer tensor buffer where:

- **Value 0**: Solid cell (closed cell) - no flow allowed, acts as a wall
- **Value 1**: Fluid cell (open cell) - normal flow
- **Value 2**: (Internal use) Fluid cell adjacent to solid boundary - automatically computed

### Important Usage Notes

When using `binary_media`, boundary conditions must be specified for TWO distinct types of boundaries:

1. **Domain Edge Boundaries**: Use standard boundary specifiers (`top`, `bottom`, `left`, `right`, `front`, `back`) for the computational domain edges
2. **Internal Solid Boundaries**: Use `boundary = wall` for the solid obstacles/structures defined by binary_media

### Example with Binary Media

```
[TensorComputes]
  [Initialize]
    [binary_media]
        type = ParsedCompute
        buffer = binary_media
        expression = '...' # expression that defined the media
        extra_symbols = true
        is_integer = true
    []
  []

  [Boundary]
    # Domain edge boundary (top wall of domain)
    [top]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = top
    []

    # Internal solid boundary (obstacle surface)
    [obstacle_wall]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = wall  # Critical: use 'wall' for binary media obstacles
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  binary_media = binary_media
  substeps = 100
[]
```

Without `binary_media`, all cells default to value 1 (fluid), and only domain edge boundaries need to be specified.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=Problem

!syntax parameters /Problem/LatticeBoltzmannProblem

!syntax inputs /Problem/LatticeBoltzmannProblem

!syntax children /Problem/LatticeBoltzmannProblem
