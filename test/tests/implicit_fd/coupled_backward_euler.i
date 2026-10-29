# The coupled pair of steady_transient.i,
#
#   c_t = D Laplacian(c) + k phi,   -Laplacian(phi) = s c,
#
# with the nonlinear solve, which iterates the step to the fixed point of the fully implicit
# backward Euler step of both fields:
#
#   (1/dt - D Lap_h) c^{n+1} = c^n / dt + k phi^{n+1},   -Lap_h phi^{n+1} = s c^{n+1}.
#
# For the mode sin(2 pi x) sin(2 pi y), an eigenvector of -Lap_h with eigenvalue lambda,
# phi^{n+1} = (s / lambda) c^{n+1}, so c^n = r^n m with
#
#   r = (1/dt) / (1/dt + D lambda - k s / lambda)
#
# and phi^n = (s / lambda) r^n m for n >= 1 (phi^0 = (s / lambda) m too). The lagged scheme of
# steady_transient.i takes a different, two-step recurrence. The solver tolerances are tight, so
# the errors are round-off: they test the fixed point, not where a stopping test lands.

D = 0.1
k = -50
s = 1
dt = 0.01
n = 32

lambda = '${fparse 2 * n^2 * (2 - 2 * cos(2 * pi / n))}'
r = '${fparse (1 / dt) / (1 / dt + D * lambda - k * s / lambda)}'

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
  [c]
  []
  [phi]
  []
  [dcdt]
  []
  [lap_c]
  []
  [lap_phi]
  []
  [src_c]
  []
  [src_phi]
  []
  [gold_c]
  []
  [gold_phi]
  []
  [diff_c]
  []
  [diff_phi]
  []
[]

[TensorComputes]
  [Initialize]
    [c_ic]
      type = ParsedCompute
      buffer = c
      expression = 'sin(2*pi*x) * sin(2*pi*y)'
      extra_symbols = true
    []
    [phi_ic]
      type = ParsedCompute
      buffer = phi
      expression = 'p0 * sin(2*pi*x) * sin(2*pi*y)'
      constant_names = 'p0'
      constant_expressions = '${fparse s / lambda}'
      extra_symbols = true
    []
  []
  [Solve]
    # the sources, evaluated by the nonlinear solve at every iterate of both fields
    [src_c_compute]
      type = ParsedCompute
      buffer = src_c
      expression = 'k * phi'
      constant_names = 'k'
      constant_expressions = '${k}'
      inputs = 'phi'
    []
    [src_phi_compute]
      type = ParsedCompute
      buffer = src_phi
      expression = 's * c'
      constant_names = 's'
      constant_expressions = '${s}'
      inputs = 'c'
    []
    [dcdt_op]
      type = TimeDerivativeOperator
      buffer = dcdt
      input = c
    []
    [lap_c_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_c
      input = c
      factor = '${fparse -D}'
    []
    [lap_phi_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_phi
      input = phi
      factor = -1
    []
  []
  [Postprocess]
    [gold_c_compute]
      type = ParsedCompute
      buffer = gold_c
      expression = 'r^(t / dt) * sin(2*pi*x) * sin(2*pi*y)'
      constant_names = 'r dt'
      constant_expressions = '${r} ${dt}'
      extra_symbols = true
    []
    [gold_phi_compute]
      type = ParsedCompute
      buffer = gold_phi
      expression = 'p * r^(t / dt) * sin(2*pi*x) * sin(2*pi*y)'
      constant_names = 'r dt p'
      constant_expressions = '${r} ${dt} ${fparse s / lambda}'
      extra_symbols = true
    []
    [diff_c_compute]
      type = ParsedCompute
      buffer = diff_c
      expression = 'abs(c - gold_c)'
      inputs = 'c gold_c'
    []
    [diff_phi_compute]
      type = ParsedCompute
      buffer = diff_phi
      expression = 'abs(phi - gold_phi)'
      inputs = 'phi gold_phi'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  # the swapped order is the cli_args buffer='phi c' operators='lap_phi_op; dcdt_op lap_c_op'
  # source='src_phi src_c'
  buffer = 'c phi'
  operators = 'dcdt_op lap_c_op; lap_phi_op'
  source = 'src_c src_phi'
  linear_solver = TORCH_PCG
  l_tol = 1e-13
  nonlinear_solve = true
  nl_rel_tol = 1e-13
[]

[Postprocessors]
  [error_c]
    type = TensorExtremeValuePostprocessor
    buffer = diff_c
    value_type = MAX
  []
  [error_phi]
    type = TensorExtremeValuePostprocessor
    buffer = diff_phi
    value_type = MAX
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
  []
  # zero up to round-off: the corrections keep the mean of c the summed equations fix
  [mean_c]
    type = TensorAveragePostprocessor
    buffer = c
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
