# TensorIntegralChangePostprocessor

!alert construction title=Undocumented Class
The TensorIntegralChangePostprocessor has not been documented. The content listed below should be used as a starting point for
documenting the class, which includes the typical automatic documentation associated with a
MooseObject; however, what is contained is ultimately determined by what is necessary to make the
documentation clear for users.

!syntax description /UserObjects/TensorIntegralChangePostprocessor

## Overview

Reports $\int |u - u_\text{old}|\,dV$ for the buffer $u$ selected with
[!param](/Postprocessors/TensorIntegralChangePostprocessor/buffer), where $u_\text{old}$ is its
previous state. The tensor solver advances the state every substep, so this is the change over the
last substep. Before the first old state exists (the first time step) it reports $\int |u|\,dV$.
The integral runs over each rank's owned cells (no ghost layers) and is summed over the ranks.
It is useful for detecting a steady state, e.g. with a `Terminator`.

## Example Input File Syntax

!listing test/tests/cahnhilliard/cahnhilliard.i block=Postprocessors/delta_int_c

!syntax parameters /UserObjects/TensorIntegralChangePostprocessor

!syntax inputs /UserObjects/TensorIntegralChangePostprocessor

!syntax children /UserObjects/TensorIntegralChangePostprocessor
