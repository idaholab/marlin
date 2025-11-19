# TensorProblem

!syntax description /Problem/TensorProblem

`TensorProblem` is the main problem class in Marlin. It adds the concepts of a [`Domain`](/DomainAction.md),
 `TensorBuffers`, `TensorComputes`, and `TensorOutputs`.

The TensorProblem object handles the execution of `TensorComputes`, schedules `TensorOutputs`, and supports
fast mapping for tensor buffers onto conforming meshes using the [!param](/TensorBuffers/PlainTensorBuffer/map_to_aux_variable) option.

## Overview

Coordinator for Marlin tensor simulations: owns the Domain, schedules `TensorComputes`, and manages
`TensorOutputs`. Supports fast projection of buffers to mesh variables via
[!param](/TensorBuffers/PlainTensorBuffer/map_to_aux_variable).

## Example Input File Syntax

!listing test/tests/gradient/gradient_square.i block=Problem

!syntax parameters /Problem/TensorProblem

!syntax inputs /Problem/TensorProblem

!syntax children /Problem/TensorProblem
