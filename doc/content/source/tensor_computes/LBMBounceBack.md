# LBMBounceBack

!syntax description /TensorComputes/Solve/LBMBounceBack

This compute object implements simple bounce-back rule on boundaries for Lattice Boltzmann simulations. Boundary can be on the left, right, top, bottom, front and back as well as wall. Wall boundary refers to any solid object the fluid cannot penetrate.

## Overview

Imposes no-penetration by reflecting incoming distributions into their opposite directions at the
selected boundary. Supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. Corner exclusion on each axis can be enabled to avoid double-applying rules.

## Usage with Binary Media

When using `binary_media` in LatticeBoltzmannProblem, boundary conditions must be specified for both:

1. **Domain edges**: Use `boundary = left/right/top/bottom/front/back` for computational domain boundaries
2. **Internal solid obstacles**: Use `boundary = wall` for solids defined by binary_media (where binary_media = 0)

The `wall` boundary type automatically detects fluid cells adjacent to solid regions in the binary media and applies bounce-back at the fluid-solid interface. This allows simulation of complex internal geometries without explicitly specifying boundary locations.

## Example Input File Syntax

!listing
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
!listing-end

!syntax parameters /TensorComputes/Solve/LBMBounceBack

!syntax inputs /TensorComputes/Solve/LBMBounceBack

!syntax children /TensorComputes/Solve/LBMBounceBack
