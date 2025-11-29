# LatticeBoltzmannProblem

!syntax description /Problem/LatticeBoltzmannProblem

The problem object derived from TensorProblem `(/TensorProblem.md)` is used to solve Lattice Boltzmann simulations. It works in the same way as TensorProblem, with a few extra features. It adds stencil and boundary condition object and modifies the initialization and execution order in favor of LBM.

## Overview

Problem driver specialized for Lattice Boltzmann simulations: manages stencils, distribution buffers,
macroscopic fields, and substepping between collision and streaming. Provides access to LBM
constants like `c_s` and time step for objects that require them.

## Scalar Constants

The `LatticeBoltzmannProblem` allows defining global scalar constants that can be retrieved by other objects (like collision operators, force computations, or boundary conditions) using their names. This is useful for defining physical parameters like relaxation times, reference densities, or external forces in one place.

Use [!param](/Problem/LatticeBoltzmannProblem/scalar_constant_names) to provide a list of names and [!param](/Problem/LatticeBoltzmannProblem/scalar_constant_values) to provide the corresponding values.

### Example

```
[Problem]
  type = LatticeBoltzmannProblem
  scalar_constant_names = 'rho0  tau_f  g'
  scalar_constant_values = '1.0   0.7    0.0001'
  substeps = 100
[]
```

These constants can then be referenced in other blocks. For example:

```
[TensorComputes]
  [Solve]
    [collision]
      type = LBMBGKCollision
      ...
      tau0 = 'tau_f'  # Reference the constant defined in Problem
    []
  []
[]
```

## Binary Media for Complex Geometries

The `binary_media` parameter enables simulation of flow through complex solid geometries (porous media, obstacles, etc.). It accepts an integer tensor buffer where:

- **Value 0**: Solid cell (closed cell) - no flow allowed, acts as a wall
- **Value >=1**: Fluid cell (open cell) - normal flow

### Important Usage Notes

When using `binary_media`, boundary conditions must be specified for 3 distinct types of boundaries:

1. **Domain Edge Boundaries**: Use standard boundary specifiers (`top`, `bottom`, `left`, `right`, `front`, `back`) for the computational domain edges
2. **Internal Solid Boundaries**: Use `boundary = wall` for the solid obstacles/structures defined by binary_media
3. **Regional Solid Boundaries**: Similar to the previous one, but user can define regions by making the parts of the binary media equal to an integer greater than 1. Then by using `boundary = regional` and `regiona_id = 2` a specialized BC can applied in that region. This is helpful in defining heat source with Dirichlet or Neumann boundary conditions.

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

    # Internal solid boundary that acts a wall and a heat source
    [heat_source]
      type = LBMNeumannBC
      buffer = g
      f_old = gpc
      feq = geq
      velocity = velocity
      rho = T
      gradient = 0.001
      region_id = 2
      boundary = regional
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
