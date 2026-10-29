# BCs System

This system is used to specify generic, non-LBM boundary conditions for the real-space
finite-difference solve path (as opposed to the LBM-specific lattice boundary conditions
under the [Boundary](syntax/TensorComputes/Boundary/index.md) system).

## Overview

Each object fills the ghost cells of one buffer on one face (`left`, `right`, `bottom`,
`top`, `back`, `front`) of a non-periodic axis, before any compute that reads the ghost
cells runs. All of them impose the cell-centred Robin condition

!equation
a u + b \frac{\partial u}{\partial n} = c

on the face, with $n$ the outward normal, $u$ approximated by the average of the
boundary-adjacent cell and its ghost, and $\partial u / \partial n$ by their difference
over the grid spacing. `TensorDirichletBC` ($a = 1$, $b = 0$) and `TensorNeumannBC`
($a = 0$, $b = 1$) are presets of `TensorRobinBC`. Every non-periodic face of a buffer
that is read with ghost cells needs a boundary condition.

The coefficients $a$ and $b$ are constants. The value $c$ (`value` and `flux` in the
presets) is a constant, or varies along the face, given by at most one of

- a function (`c_function`, `value_function`, `flux_function`), evaluated at the face
  position and the cell centres along the face, at the time of the solve: the current
  time for explicit solvers, the new time for [ImplicitFDSolver.md], or
- a buffer (`c_buffer`, `value_buffer`, `flux_buffer`), whose first owned cell layer next
  to the face holds $c$. The compute that writes it must evaluate $c$ at the face
  position, not at the cell centres. Its values are those the compute last wrote: for
  [ImplicitFDSolver.md], whose explicit computes run at the start of the substep, they lag
  the new time by one time step. A buffer that no compute or initial condition writes is
  an error. The compute runs before any compute that reads the variable with ghost cells.

A stencil reaching $L$ cells across the face (e.g. two for the 5-point Laplacian) needs $L$ ghost
layers. Ghost layer $l = 0, \dots, L-1$ is filled from the same condition applied to it and its
mirror, owned cell $l$ from the face, which are $(2l + 1) \Delta x$ apart: the even mirror for a
zero-flux face and the odd mirror about $c$ for a Dirichlet face. Every rank next to the face must
own at least $L$ cells normal to it, and $a/2 + b/((2l + 1) \Delta x)$ must be nonzero for each
layer; both are checked at setup.

## Example Input File Syntax

A Dirichlet condition on the left face and a Neumann condition on the right face of a
domain that is periodic in $y$:

!listing test/tests/real_space/neumann_linear.i block=Domain TensorComputes/BCs

Values that vary along the face, from functions and from buffers:

!listing test/tests/real_space/bc_varying_laplacian.i block=TensorComputes/BCs

!syntax list /TensorComputes/BCs objects=True actions=False subsystems=False

!syntax list /TensorComputes/BCs objects=False actions=False subsystems=True

!syntax list /TensorComputes/BCs objects=False actions=True subsystems=False
