# LBMBounceBack

!syntax description /TensorComputes/Boundary/LBMBounceBack

This compute object implements simple bounce-back rule on boundaries for Lattice Boltzmann simulations. Boundary can be on the left, right, top, bottom, front and back as well as wall and regional. Wall boundary refers to any solid object the fluid cannot penetrate.

## Overview

Imposes no-penetration by reflecting incoming distributions into their opposite directions at the
selected boundary. Supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` and `regional` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. Corner exclusion on each axis can be enabled to avoid double-applying rules.
Choose faces with [!param](/TensorComputes/Boundary/LBMFixedZerothOrderBC/boundary).
Exclude corners with [!param](/TensorComputes/Boundary/LBMFixedZerothOrderBC/exclude_corners_x) set to `True`
Old state buffer is the post collision distribution [!param](/TensorComputes/Boundary/LBMFixedZerothOrderBC/f_old).

## Usage with Binary Media

When using `binary_media` in LatticeBoltzmannProblem, boundary conditions must be specified for both:

1. **Domain edges**: Use `boundary = left/right/top/bottom/front/back` for computational domain boundaries
2. **Internal solid obstacles**: Use `boundary = wall` for solids defined by binary_media (where binary_media = 0)

The `wall` boundary type automatically detects fluid cells adjacent to solid regions in the binary media and applies bounce-back at the fluid-solid interface. This allows simulation of complex internal geometries without explicitly specifying boundary locations.

> **Note**: The most flexible method to generate binary media is to use custom Python scripts and the load it as a tensor buffer from HDF5. The examples of these scripts can be found in `examples/lbm/`. Do not forget to pass the binary media buffer into the Problem block. See the example below.

## Example Input File Syntax

!listing
[TensorBuffers]
  [binary_media]
    type = LBMTensorBuffer
    file = binary_media.h5
    is_integer = true
    buffer_type = ms
  []
[]

[TensorComputes]
  [Boundary]
    [bb]
      type = LBMBounceBack
      buffer = f
      f_old = f_post_collision
      boundary = 'left right top bottom'
      exclude_corners_x = true
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 100
  binary_media = binary_media
[]
!listing-end

!syntax parameters /TensorComputes/Boundary/LBMBounceBack

!syntax inputs /TensorComputes/Boundary/LBMBounceBack

!syntax children /TensorComputes/Boundary/LBMBounceBack
