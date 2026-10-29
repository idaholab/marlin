# TensorNeumannBC

!syntax description /TensorComputes/BCs/TensorNeumannBC

## Overview

Imposes $\partial u / \partial n =$ [!param](/TensorComputes/BCs/TensorNeumannBC/flux) on one
face of a non-periodic direction, with $n$ the outward normal. It is [TensorRobinBC.md] with
$a = 0$ and $b = 1$. The flux can vary along the face, from
[!param](/TensorComputes/BCs/TensorNeumannBC/flux_function) or
[!param](/TensorComputes/BCs/TensorNeumannBC/flux_buffer) (see the
[BCs system](syntax/TensorComputes/BCs/index.md)).

## Example Input File Syntax

!listing test/tests/real_space/neumann_linear.i block=TensorComputes/BCs

!syntax parameters /TensorComputes/BCs/TensorNeumannBC

!syntax inputs /TensorComputes/BCs/TensorNeumannBC

!syntax children /TensorComputes/BCs/TensorNeumannBC
