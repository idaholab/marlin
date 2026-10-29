# TensorRobinBC

!syntax description /TensorComputes/BCs/TensorRobinBC

## Overview

Imposes $a u + b \partial u / \partial n = c$ on one face of a non-periodic direction, with $n$
the outward normal. See the [BCs system](syntax/TensorComputes/BCs/index.md) for the
discretization and for $c$ varying along the face
([!param](/TensorComputes/BCs/TensorRobinBC/c_function) or
[!param](/TensorComputes/BCs/TensorRobinBC/c_buffer)). $a/2 + b/\Delta x$ must be nonzero, with
$\Delta x$ the grid spacing normal to the face. For $a < 0$ the ghost value amplifies the
boundary-adjacent cell, and explicit time integration diverges. [ImplicitFDSolver.md] supports a
Robin face with the 3-point stencil only, and rejects one with $a b < 0$ (see that page).

## Example Input File Syntax

!listing test/tests/real_space/robin_steady_state.i block=TensorComputes/BCs

!syntax parameters /TensorComputes/BCs/TensorRobinBC

!syntax inputs /TensorComputes/BCs/TensorRobinBC

!syntax children /TensorComputes/BCs/TensorRobinBC
