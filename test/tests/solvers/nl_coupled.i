#
# Simple Cahn-Hilliard solve on a 2D grid.
#

[Domain]
  dim = 2
  nx = 150
  ny = 150
  xmax = '${fparse pi*2}'
  ymax = '${fparse pi*2}'
  mesh_mode = DUMMY
[]

[GlobalParams]
  constant_names = 'A B'
  constant_expressions = '1 3.5'
[]

[TensorComputes]
  [Initialize]
    [u]
      type = ParsedCompute
      buffer = u
      extra_symbols = true
      expression = 'sin(x)*sin(y)'
      expand = REAL
    []
    [v]
      type = ParsedCompute
      buffer = v
      extra_symbols = true
      expression = 'cos(x)*cos(y)'
      expand = REAL
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
    []

    # precompute fixed factors for the solve
    [D1]
      type = ReciprocalLaplacianFactor
      factor = 1e-2
      buffer = D1
    []
    [D2]
      type = ReciprocalLaplacianFactor
      factor = 1e-3
      buffer = D2
    []
  []

  [Solve]
    [u_bar]
      type = ForwardFFT
      buffer = u_bar
      input = u
    []
    [v_bar]
      type = ForwardFFT
      buffer = v_bar
      input = v
    []

    [Du]
      type = ParsedCompute
      buffer = Du
      expression = 'D2*v_bar'
      inputs = 'D2 v_bar'
    []
    [Dv]
      type = ParsedCompute
      buffer = Dv
      expression = 'D2*u_bar'
      inputs = 'D2 u_bar'
    []
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'u v'
  reciprocal_buffer = 'u_bar v_bar'
  linear_reciprocal = 'D1 D1'
  nonlinear_reciprocal = 'Du Dv'
  substeps = ${ss}
  corrector_steps = ${cs}
  predictor_order = ${order}
  corrector_order = ${order}
[]

[Problem]
  type = TensorProblem
[]

[Postprocessors]
  [u_min]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
  []
  [u_max]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
  [v_min]
    type = TensorExtremeValuePostprocessor
    buffer = v
    value_type = MIN
  []
  [v_max]
    type = TensorExtremeValuePostprocessor
    buffer = v
    value_type = MAX
  []
  [U]
    type = TensorIntegralPostprocessor
    buffer = u
  []
  [V]
    type = TensorIntegralPostprocessor
    buffer = v
  []
[]

[Executioner]
  type = Transient
  num_steps = 25
  dt = 10
[]

[Outputs]
  file_base = nl_coupled_${ss}_${cs}_${order}
  csv = true
[]
