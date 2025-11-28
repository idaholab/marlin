# LBMFixedZerothOrderBC

!syntax description /TensorComputes/Boundary/LBMFixedZerothOrderBC

LBMFixedZerothOrderBC implements Zou-He pressure boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils.

## Overview

Enforces first\-order accurate macroscopic pressure at selected domain faces via Zou\-He formulas.
Choose faces with [!param](/TensorComputes/Boundary/LBMFixedZerothOrderBC/boundary) and provide macroscopic fields as
required by the implementation.

## Example Input File Syntax

!listing test/tests/lbm/vertical_density_bcs.i block=TensorComputes/Boundary/top
!listing test/tests/lbm/vertical_density_bcs.i block=TensorComputes/Boundary/bottom

!syntax parameters /TensorComputes/Boundary/LBMFixedZerothOrderBC
