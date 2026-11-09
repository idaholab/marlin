# FiniteDifferenceDivGrad applied explicitly to a smooth u with a smooth coefficient D, with
# Dirichlet faces at x = 0, 1 and Neumann faces at y = 0, 1.5. The postprocessors of div(D grad u)
# are compared with an independent NumPy evaluation of the same discretisation (harmonic or
# arithmetic face mean, the boundary cell's D on a domain face).
#
#   u = sin(2x) cos(y) + x y,  D = 1 + x/2 + 0.3 cos(3y)

face_mean = harmonic

[Domain]
  dim = 2
  nx = 32
  ny = 24
  xmax = 1
  ymax = 1.5
  parallel_mode = REAL_SPACE
[]

[Functions]
  [u_func]
    type = ParsedFunction
    expression = 'sin(2*x)*cos(y) + x*y'
  []
  # du/dn on y = 0 (outward normal -y) and on y = 1.5
  [bottom_flux]
    type = ParsedFunction
    expression = '-x'
  []
  [top_flux]
    type = ParsedFunction
    expression = '-sin(2*x)*sin(y) + x'
  []
[]

[TensorComputes]
  [Initialize]
    [u]
      type = MooseFunctionTensor
      buffer = u
      function = u_func
    []
    [D]
      type = ParsedCompute
      buffer = D
      extra_symbols = true
      expression = '1 + x/2 + 0.3*cos(3*y)'
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value_function = u_func
    []
    [right]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value_function = u_func
    []
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux_function = bottom_flux
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = top_flux
    []
  []
  [Solve]
    [divgrad]
      type = FiniteDifferenceDivGrad
      buffer = divgrad
      input = u
      coefficient = D
      face_mean = ${face_mean}
      factor = 2
    []
    [square]
      type = ParsedCompute
      buffer = square
      expression = 'divgrad^2'
      inputs = divgrad
    []
  []
[]

[Postprocessors]
  [integral]
    type = TensorIntegralPostprocessor
    buffer = divgrad
  []
  [integral_square]
    type = TensorIntegralPostprocessor
    buffer = square
  []
  [max]
    type = TensorExtremeValuePostprocessor
    buffer = divgrad
    value_type = MAX
  []
  [min]
    type = TensorExtremeValuePostprocessor
    buffer = divgrad
    value_type = MIN
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
