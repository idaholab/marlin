# ComputeReynoldsNumber

!syntax description /UserObjects/ComputeReynoldsNumber

The ComputeReynoldsNumber objects acts on speed buffer. The inputs tau and diameter are relaxation parameter and the representative scale respectively. The return value is Reynolds number.

## Overview

Computes the Reynolds number `Re = U D / \nu` from a speed buffer and user\-provided length scale.
Provide the speed via [!param](/Postprocessors/ComputeReynoldsNumber/buffer), relaxation parameter
through [!param](/Postprocessors/ComputeReynoldsNumber/tau), and the characteristic
diameter with [!param](/Postprocessors/ComputeReynoldsNumber/diameter).

## Example Input File Syntax

!listing examples/lbm/Karman-vortex/cylinder.i block=Postprocessors/reynolds

!syntax parameters /UserObjects/ComputeReynoldsNumber

!syntax inputs /UserObjects/ComputeReynoldsNumber

!syntax children /UserObjects/ComputeReynoldsNumber
