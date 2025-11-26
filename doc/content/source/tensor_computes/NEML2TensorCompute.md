# NEML2TensorCompute

!syntax description /TensorComputes/Solve/NEML2TensorCompute

Use a NEML2 model to perform a tensor compute.

## Overview

Maps Marlin buffers to NEML2 model inputs, executes the model, and maps model outputs back to Marlin
buffers. Configure the NEML2 input file and model via
[!param](/TensorComputes/Solve/NEML2TensorCompute/neml2_input_file) and
[!param](/TensorComputes/Solve/NEML2TensorCompute/neml2_model). Use
[!param](/TensorComputes/Solve/NEML2TensorCompute/marlin_inputs),
[!param](/TensorComputes/Solve/NEML2TensorCompute/neml2_inputs),
[!param](/TensorComputes/Solve/NEML2TensorCompute/neml2_outputs), and
[!param](/TensorComputes/Solve/NEML2TensorCompute/marlin_outputs) to define the mappings.

## Example Input File Syntax

!listing test/tests/neml2/scalar.i block=TensorComputes/Initialize/C

!syntax parameters /TensorComputes/Solve/NEML2TensorCompute

!syntax inputs /TensorComputes/Solve/NEML2TensorCompute

!syntax children /TensorComputes/Solve/NEML2TensorCompute
