[Domain]
  dim = 2
  nx = 200
  ny = 200
  xmax = 200
  ymax = 200

  device_names = 'cuda'

  mesh_mode = DOMAIN
[]


[TensorBuffers]
  [c]
    map_to_aux_variable = c
  []
  [cbar]
  []
  [mu]
    # map_to_aux_variable = mu
  []
  [mubar]
  []
  [Mbarmubar]
  []
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
  # postprocessing
  [F]
  []
  [Fgrad]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'c0+epsilon*(cos(0.105*x)*cos(0.11*y)+(cos(0.13*x)*cos(0.087*y))^2+cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y))'
      constant_names = 'c0 epsilon'
      constant_expressions = '0.5 0.01'
    []
    [Mbar]
      type = ReciprocalLaplacianFactor
      factor = 5 # Mobility
      buffer = Mbar
    []
    [kappabarbar]
      type = ReciprocalLaplacianSquareFactor
      factor = -10 # -kappa*M
      buffer = kappabarbar
    []
  []

  [Solve]
    [mu]
      type = ParsedCompute
      buffer = mu
      expression = 'rho_s*(c-c_alpha)^2*(c_beta-c)^2'
      constant_names =       'rho_s c_alpha c_beta'
      constant_expressions = '5     0.3     0.7'
      derivatives = c
      inputs = c
    []
    [mubar]
      type = ForwardFFT
      buffer = mubar
      input = mu
    []
    [Mbarmubar]
      type = ParsedCompute
      buffer = Mbarmubar
      expression = 'Mbar*mubar'
      inputs = 'Mbar mubar'
    []
    [cbar]
      type = ForwardFFT
      buffer = cbar
      input = c
    []
  []

  [Postprocess]
    [Fgrad]
      type = FFTGradientSquare
      buffer = Fgrad
      input = c
      factor = 1 # kappa/2
    []
    [F]
      type = ParsedCompute
      buffer = F
      expression = 'rho_s * (c-c_alpha)^2 * (c_beta-c)^2 + Fgrad'
      constant_names =       'rho_s c_alpha c_beta'
      constant_expressions = '5     0.3     0.7'
      inputs = 'c Fgrad'
    []
  []
[]

[UserObjects]
  [terminator]
    type = Terminator
    expression = change<1e-4
  []
[]

[TensorTimeIntegrators]
  [c]
    type = FFTSemiImplicit
    buffer = c
    history_size = 1
    reciprocal_buffer = cbar
    linear_reciprocal = kappabarbar
    nonlinear_reciprocal = Mbarmubar
  []
[]

[AuxVariables]
  # [mu]
  #   family = MONOMIAL
  #   order = CONSTANT
  # []
  [c]
    # family = MONOMIAL
    # order = CONSTANT
  []
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [F]
    type = TensorIntegralPostprocessor
    buffer = F
  []
  [change]
    type = TensorIntegralChangePostprocessor
    buffer = c
  []
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 1000
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.1
    dt = 1
  []
  dtmax = 300
[]

[Outputs]
  exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
