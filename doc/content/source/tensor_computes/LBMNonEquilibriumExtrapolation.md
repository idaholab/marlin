# LBMNonEquilibriumExtrapolation

!syntax description /TensorComputes/Boundary/LBMNonEquilibriumExtrapolation

Implements the Non-Equilibrium Extrapolation Method (NEEM) boundary condition for Lattice Boltzmann simulations (Guo et al., 2002).

## Overview

This boundary condition reconstructs the distribution function at the boundary as

$$f_i(\mathbf{x}_b) = f_i^{eq}(\rho_b, \mathbf{u}_b) + f_i^{neq}(\mathbf{x}_{n1})$$

where $f_i^{eq}$ is the equilibrium distribution computed from the prescribed/extrapolated macroscopic
quantities at the boundary, and $f_i^{neq} = f_i - f_i^{eq}$ is the non-equilibrium part extrapolated
from interior nodes.

Two prescription modes are available via
[!param](/TensorComputes/Boundary/LBMNonEquilibriumExtrapolation/prescribe_type):

- `velocity` (default): The velocity is prescribed at the boundary and the density is extrapolated from the interior.
- `density`: The density is prescribed and the velocity is extrapolated from the interior.

The extrapolation order is controlled by
[!param](/TensorComputes/Boundary/LBMNonEquilibriumExtrapolation/order):

- `first` (default): Uses one interior neighbor. $q_b = q_{n1}$ where $q$ is the extrapolated quantity.
- `second`: Uses two interior neighbors. $q_b = 2 q_{n1} - q_{n2}$ for linear extrapolation of the non-prescribed quantity and the non-equilibrium part.

All macroscopic quantities (density, velocity) and the equilibrium distribution are computed
on-the-fly from the current (post-streaming) population, ensuring they are always consistent
with the latest state of the distribution function.

It supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`).

## Example Input File Syntax

!listing test/tests/lbm/nee_2d_channel.i block=TensorComputes/Boundary/left

!listing test/tests/lbm/nee_2d_channel.i block=TensorComputes/Boundary/right

!listing test/tests/lbm/nee_2d_all_walls.i block=TensorComputes/Boundary/top

!syntax parameters /TensorComputes/Boundary/LBMNonEquilibriumExtrapolation

!syntax inputs /TensorComputes/Boundary/LBMNonEquilibriumExtrapolation

!syntax children /TensorComputes/Boundary/LBMNonEquilibriumExtrapolation
