# TensorDirichletBC

!syntax description /TensorComputes/BCs/TensorDirichletBC

## Overview

Imposes $u =$ [!param](/TensorComputes/BCs/TensorDirichletBC/value) on one face of a
non-periodic direction. It is [TensorRobinBC.md] with $a = 1$ and $b = 0$. The value can vary
along the face, from [!param](/TensorComputes/BCs/TensorDirichletBC/value_function) or
[!param](/TensorComputes/BCs/TensorDirichletBC/value_buffer) (see the
[BCs system](syntax/TensorComputes/BCs/index.md)).

## Example Input File Syntax

!listing test/tests/real_space/dirichlet_linear.i block=TensorComputes/BCs

!syntax parameters /TensorComputes/BCs/TensorDirichletBC

!syntax inputs /TensorComputes/BCs/TensorDirichletBC

!syntax children /TensorComputes/BCs/TensorDirichletBC
