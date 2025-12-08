[Domain]
  dim = 2
  nx = 80
  ny = 60
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
  xmin = -1
  ymin = -1
  xmax = 1
  ymax = 1
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      expression = 'exp((-x^2-y^2)*100)'
      extra_symbols = true
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
    []
  []
[]

[TensorSolver]
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 100
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [U]
    type = TensorIntegralPostprocessor
    buffer = u
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[TensorOutputs]
  # active = ''
  [u]
    type = XDMFTensorOutput
    buffer = 'u dudt'
    enable_hdf5 = true
  []
[]

[Executioner]
  type = Transient
  dt = 1e-2
  num_steps = 100
[]

[Outputs]
  [out]
    type = CSV
  []
[]
