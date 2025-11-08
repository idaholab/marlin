[Domain]
  dim = 2
  nx = 4
  ny = 4
  xmax = 1
  ymax = 1

  device_names = 'cuda:0'

  mesh_mode = DOMAIN
[]

[TensorBuffers]
  # variables
  [n1]
  []
  [n2]
  []
  [n3]
  []

  [n1_bar]
  []
  [n2_bar]
  []
  [n3_bar]
  []

  [mu_n1]
  []
  [mu_n2]
  []
  [mu_n3]
  []

  [mu_n1_bar]
  []
  [mu_n2_bar]
  []
  [mu_n3_bar]
  []

  [Lbar] # zero
  []
[]

[TensorComputes]
  [Initialize]
    [Lbar]
      type = ConstantReciprocalTensor
      buffer = Lbar
      real = 0
      imaginary = 0
    []

    [n1]
      type = ConstantTensor
      buffer = n1
      real = 1
    []
    [n2]
      type = ConstantTensor
      buffer = n2
      real = 1
    []
    [n3]
      type = ConstantTensor
      buffer = n3
      real = 1
    []

    [mu_n1]
      type = ConstantTensor
      buffer = mu_n1
      real = 1
    []
  []

  [Solve]
    [mu_n2]
      type = ParsedCompute
      buffer = mu_n2
      expression = 'n3'
      inputs = n3
    []
    [mu_n3]
      type = ParsedCompute
      buffer = mu_n3
      expression = 'n3*n3'
      inputs = n3
    []

    [mu_n1_bar]
      type = ForwardFFT
      buffer = mu_n1_bar
      input = mu_n1
    []
    [mu_n2_bar]
      type = ForwardFFT
      buffer = mu_n2_bar
      input = mu_n2
    []
    [mu_n3_bar]
      type = ForwardFFT
      buffer = mu_n3_bar
      input = mu_n3
    []

    [n1_bar]
      type = ForwardFFT
      buffer = n1_bar
      input = n1
    []
    [n2_bar]
      type = ForwardFFT
      buffer = n2_bar
      input = n2
    []
    [n3_bar]
      type = ForwardFFT
      buffer = n3_bar
      input = n3
    []
  []
[]

[TensorSolver]
  # type = SecantSolver
  type = BroydenSolver
  substeps = 1
  max_iterations = 10
  damping = 0.5
  buffer = 'n1 n2 n3'
  tolerance = 1e-5
  dt_epsilon = 1e-5
  reciprocal_buffer = 'n1_bar n2_bar n3_bar'
  linear_reciprocal = 'Lbar   Lbar   Lbar'
  nonlinear_reciprocal = 'mu_n1_bar mu_n2_bar mu_n3_bar'
  verbose = true
[]

[Postprocessors]
  [n1]
    type = TensorAveragePostprocessor
    buffer = n1
    execute_on = 'TIMESTEP_END'
  []
  [n2]
    type = TensorAveragePostprocessor
    buffer = n2
    execute_on = 'TIMESTEP_END'
  []
  [n3]
    type = TensorAveragePostprocessor
    buffer = n3
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 1
  dt = 1e-2
[]

[Outputs]
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
