[Domain]
  device_names = "cuda:1 cuda:0 cpu"
  device_weights = "10 10 1"

  parallel_mode = FFT_SLAB

  dim = 2
  nx = 400
  ny = 400
  xmax = ${fparse pi*2*30}
  ymax = ${fparse pi*2*30}
[]


# [Mesh]
#   type = GeneratedMesh
#   dim =1
# []


[TensorBuffers]
  [psi]
    # map_to_aux_variable = psi
  []
  [psibar]
  []
  [psi3]
  []
  [psi3bar]
  []
  # constant tensors
  [linear]
  []

  # output
  [filter]
    # map_to_aux_variable = filter
  []
  [filterbar]
  []
[]

[AuxVariables]
  [psi]
  []
  [filter]
  []
[]

[TensorComputes]
  [Initialize]
    [psi]
      type = RandomTensor
      buffer = psi
      min = 0
      max = 0.07
    []
    [linear]
      type = SwiftHohenbergLinear
      buffer = linear
      alpha = 1
      r = 0.025
    []
  []

  [Solve]
    [psi3]
      type = ParsedCompute
      buffer = psi3
      expression = "0.20*psi^2-psi^3"
      inputs = psi
    []
    [psibar]
      type = ForwardFFT
      buffer = psibar
      input = psi
    []
    [psi3bar]
      type = ForwardFFT
      buffer = psi3bar
      input = psi3
    []
  []

  [Postprocess]
    [low_pass]
      type = ParsedCompute
      buffer = filterbar
      extra_symbols = true
      expression = 'psibar * exp(-k2*10)'
      inputs = psibar
    []
    [filter]
      type = InverseFFT
      buffer = filter
      input = filterbar
    []
  []
[]

[TensorTimeIntegrators]
  [c]
    type = FFTSemiImplicit
    buffer = psi
    reciprocal_buffer = psibar
    linear_reciprocal = linear
    nonlinear_reciprocal = psi3bar
  []
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 300
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.2
    dt = 10
  []
  dtmax = 1000
[]

[Postprocessors]
  [min_psi]
    type = TensorExtremeValuePostprocessor
    buffer = psi
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_psi]
    type = TensorExtremeValuePostprocessor
    buffer = psi
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [Psi]
    type = TensorIntegralPostprocessor
    buffer = psi
  []
[]

# [TensorOutputs]
#   [xdmf]
#     type = XDMFTensorOutput
#     buffer = 'psi'
#     output_mode = 'Node'
#     enable_hdf5 = true
#   []
# []

[Outputs]
  exodus = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]

