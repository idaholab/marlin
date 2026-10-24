#
# PFHub benchmark 1a (spinodal decomposition, square periodic domain) with the
# implicit finite-difference solver. Same problem as 1a.i:
#
#   dc/dt = div(M grad(f'(c) - kappa Laplacian(c))),  f = rho_s (c - c_alpha)^2 (c_beta - c)^2
#
# Backward Euler, with the biharmonic implicit and the double-well flux
# divergence explicit:
#
#   (c - c^n)/dt + M kappa Laplacian(Laplacian(c)) = M Laplacian(f'(c^n))
#
# M = 5, kappa = 2, rho_s = 5, c_alpha = 0.3, c_beta = 0.7.
#

# substeps per time step, as in 1a_solver.i (each substep is one linear solve)
substeps = 1000

[Domain]
  dim = 2
  nx = 200
  ny = 200
  xmax = 200
  ymax = 200
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [c]
  []
  [dcdt]
  []
  [bih_c]
  []
  [mu]
  []
  [rhs]
  []
  # postprocessing
  [lap_c]
  []
  [F]
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
  []

  [Solve]
    # implicit operators
    [dcdt]
      type = TimeDerivativeOperator
      buffer = dcdt
      input = c
    []
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih_c
      input = c
      factor = 10 # M * kappa
    []

    # explicit source M Laplacian(f'(c^n))
    [explicit]
      [mu]
        type = ParsedCompute
        buffer = mu
        expression = 'rho_s*(c-c_alpha)^2*(c_beta-c)^2'
        constant_names = 'rho_s c_alpha c_beta'
        constant_expressions = '5     0.3     0.7'
        derivatives = c
        inputs = c
      []
      [rhs]
        type = FiniteDifferenceLaplacian
        buffer = rhs
        input = mu
        factor = 5 # M
      []
    []
  []

  [Postprocess]
    # On the periodic grid sum |grad c|^2 = -sum c Laplacian(c) for the 3-point
    # Laplacian, so the gradient energy (kappa/2)|grad c|^2 integrates to the same
    # total as -(kappa/2) c Laplacian(c).
    [lap_c]
      type = FiniteDifferenceLaplacian
      buffer = lap_c
      input = c
    []
    [F]
      type = ParsedCompute
      buffer = F
      expression = 'rho_s * (c-c_alpha)^2 * (c_beta-c)^2 - kappa / 2 * c * lap_c'
      constant_names = 'rho_s c_alpha c_beta kappa'
      constant_expressions = '5     0.3     0.7    2'
      inputs = 'c lap_c'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = c
  operators = 'dcdt bih'
  source = rhs
  root_compute = explicit
  substeps = ${substeps}
[]

[UserObjects]
  [terminator]
    type = Terminator
    expression = change<1e-4
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
  # cumulative wall time of the run (s)
  [wall_time]
    type = PerfGraphData
    section_name = Root
    data_type = TOTAL
  []
[]

[Problem]
  type = TensorProblem
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

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c mu'
    enable_hdf5 = true
  []
[]

[Outputs]
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
