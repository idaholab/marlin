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
      type = ConstantTensor
      buffer = v
      real = 0
    []

    # precompute fixed factors for the solve
    [Du]
      type = ReciprocalLaplacianFactor
      factor = 1e-2
      buffer = Du
    []
    [Dv]
      type = ReciprocalLaplacianFactor
      factor = 1e-3
      buffer = Dv
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

    [source_u]
      type = ParsedCompute
      buffer = source_u
      expression = 'A - (B+1)*u +u^2*v'
      inputs = 'u v'
    []
    [source_u_bar]
      type = ForwardFFT
      buffer = source_u_bar
      input = source_u
    []

    [source_v]
      type = ParsedCompute
      buffer = source_v
      expression = 'B*u - u^2*v'
      inputs = 'u v'
    []
    [source_v_bar]
      type = ForwardFFT
      buffer = source_v_bar
      input = source_v
    []
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'u v'
  reciprocal_buffer = 'u_bar v_bar'
  linear_reciprocal = 'Du Dv'
  nonlinear_reciprocal = 'source_u_bar source_v_bar'
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
  dt = 0.5
[]

[Outputs]
  file_base = diagonal_${ss}_${cs}_${order}
  csv = true
[]
