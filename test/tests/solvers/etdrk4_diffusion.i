D = 0.05
k = 1.0
ss = 1
dt = '${units 10 s }'

[Domain]
  dim = 1
  nx = 64
  xmax = '${fparse 2*pi}'
  mesh_mode = DUMMY
[]

[TensorComputes]
  [Initialize]
    [u0]
      type = ParsedCompute
      buffer = u0
      extra_symbols = true
      expression = 'sin(${k}*x)'
    []
    [u]
      type = ParsedCompute
      buffer = u
      inputs = u0
      expression = 'u0'
    []

    [L]
      type = ReciprocalLaplacianFactor
      factor = ${D}
      buffer = L
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
    []
  []

  [Solve]
    [u_bar]
      type = ForwardFFT
      buffer = u_bar
      input = u
    []
    [u_exact]
      type = ParsedCompute
      buffer = u_exact
      inputs = u0
      extra_symbols = true
      expression = 'u0*exp(-${D}*${k}^2*t)'
    []
    [u_diff_sq]
      type = ParsedCompute
      buffer = u_diff_sq
      inputs = 'u u_exact'
      expression = '(u - u_exact)^2'
    []
  []
[]

[TensorSolver]
  type = ETDRK4Solver
  buffer = 'u'
  reciprocal_buffer = 'u_bar'
  linear_reciprocal = 'L'
  nonlinear_reciprocal = 'zero'
  substeps = ${ss}
[]

[Problem]
  type = TensorProblem
[]

[Postprocessors]
  [mse]
    type = TensorIntegralPostprocessor
    buffer = u_diff_sq
  []
  [rmse]
    type = ParsedPostprocessor
    expression = 'sqrt(mse)'
    pp_names = 'mse'
    pp_symbols = 'mse'
  []
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = ${dt}
[]

[Outputs]
  file_base = etdrk4_diffusion_rmse
  csv = true
[]
