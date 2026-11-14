# a u_t = D Laplacian(u) with a cell-centred coefficient a > 0 in the time derivative
# (CoefficientTimeDerivativeOperator), periodic, on the step sequence of bdf2_mode.i.
# - a = 1 is bdf2_mode.i, so the gold of its exact mode recurrence applies.
# - A varying a is checked against coefficient_dt.py, an independent scipy sparse solve of the
#   same discrete steps. On a periodic domain sum_i a_i u_i is conserved (the Laplacian's
#   columns sum to zero), so mass_a stays at its first value.

D = 0.5
n = 32
stencil = 3
order = 2
# a = 1 (a constant expression would evaluate to a scalar, the ones buffer has the grid's shape)
a_expression = 'ones'

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [u]
  []
  [ones]
  []
  [a]
  []
  [a_u]
  []
  [dudt_u]
  []
  [lap_u]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      expression = 'sin(2*pi*x)*sin(2*pi*y)'
      extra_symbols = true
    []
    [ones_ic]
      type = ConstantTensor
      buffer = ones
      real = 1
    []
    [a_ic]
      type = ParsedCompute
      buffer = a
      expression = '${a_expression}'
      inputs = ones
      extra_symbols = true
    []
  []
  [Solve]
    [dudt]
      type = CoefficientTimeDerivativeOperator
      buffer = dudt_u
      input = u
      coefficient = a
      order = ${order}
    []
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = '${fparse -D}'
      stencil_width = ${stencil}
    []
  []
  [Postprocess]
    [a_u_compute]
      type = ParsedCompute
      buffer = a_u
      expression = 'a*u'
      inputs = 'a u'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  root_compute = dudt
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
  [min_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
  []
  [mass]
    type = TensorIntegralPostprocessor
    buffer = u
  []
  [mass_a]
    type = TensorIntegralPostprocessor
    buffer = a_u
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 8
  [TimeStepper]
    type = TimeSequenceStepper
    # steps 0.01, 0.01, 0.02, 0.01, 0.015, 0.0075, 0.015, 0.03
    time_sequence = '0.01 0.02 0.04 0.05 0.065 0.0725 0.0875 0.1175'
  []
[]

[Outputs]
  csv = true
[]
