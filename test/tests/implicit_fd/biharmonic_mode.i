# Backward Euler for u_t = -M kappa Laplacian(Laplacian(u)) of a single periodic
# Fourier mode. The mode is an eigenvector of the discrete Laplacian with
# eigenvalue -lambda_h, so of the discrete biharmonic with eigenvalue lambda_h^2,
# and every implicit step must multiply it by exactly
# g = 1 / (1 + dt M kappa lambda_h^2), up to the linear solver tolerance.

Mkappa = 1e-3
dt = 0.01
n = 32
stencil = 3

# Symbol of -Laplacian per axis for the wavenumber 2 pi, h = 1/n (see be_mode.i)
lambda3 = '${fparse n^2 * (2 - 2 * cos(2 * pi / n))}'
lambda5 = '${fparse n^2 * (5 / 2 - 8 / 3 * cos(2 * pi / n) + 1 / 6 * cos(4 * pi / n))}'
lambda = '${fparse 2 * if(stencil = 3, lambda3, lambda5)}'

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
  [dudt_u]
  []
  [bih_u]
  []
  [gold]
  []
  [diff]
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
  []
  [Solve]
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih_u
      input = u
      factor = ${Mkappa}
      stencil_width = ${stencil}
    []
  []
  [Postprocess]
    [gold_compute]
      type = ParsedCompute
      buffer = gold
      expression = 'sin(2*pi*x)*sin(2*pi*y) * g^(t / dt)'
      constant_names = 'g dt'
      constant_expressions = '${fparse 1 / (1 + dt * Mkappa * lambda^2)} ${dt}'
      extra_symbols = true
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - gold)'
      inputs = 'u gold'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt bih'
  root_compute = dudt
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
  [max_gold]
    type = TensorExtremeValuePostprocessor
    buffer = gold
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = ${dt}
[]

[Outputs]
  csv = true
[]
