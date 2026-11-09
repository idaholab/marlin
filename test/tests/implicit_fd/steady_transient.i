# A transient field c and a steady field phi solved by one ImplicitFDSolver, coupled through
# explicit sources:
#
#   c_t = D Laplacian(c) + k phi,   -Laplacian(phi) = s c,
#
# on the periodic unit square. Every source is computed from the fields at the start of the step,
# whatever the order of the fields in 'buffer', so a step solves
#
#   (1/dt - D Lap_h) c^{n+1} = c^n / dt + k phi^n,   -Lap_h phi^{n+1} = s c^n.
#
# The mode sin(2 pi x) sin(2 pi y) is an eigenvector of -Lap_h with eigenvalue lambda, so
# c^n = a_n m and phi^n = p_n m with
#
#   a_{n+1} = (a_n / dt + k p_n) / (1/dt + D lambda),   p_{n+1} = s a_n / lambda.
#
# phi starts at the steady solution of c^0 (p_0 = s a_0 / lambda), so p_n = s a_{n-1} / lambda
# for every n >= 1 and a_{n+1} (1/dt + D lambda) = a_n / dt + (k s / lambda) a_{n-1} for n >= 1:
# a_n = A r1^n + B r2^n with r1, r2 the roots of (1/dt + D lambda) r^2 - r / dt - k s / lambda,
# both positive for the parameters below, and A, B from a_0 = 1 and a_1. The singular phi solve
# needs a zero-mean s c, which holds because the solver sets the mean of c^{n+1} to the one the
# summed equations fix (zero here) rather than leaving the Krylov error in it.

D = 0.1
k = -50
s = 1
dt = 0.01
n = 32

lambda = '${fparse 2 * n^2 * (2 - 2 * cos(2 * pi / n))}'
alpha = '${fparse 1 / dt + D * lambda}'
beta = '${fparse k * s / lambda}'
disc = '${fparse 1 / dt^2 + 4 * alpha * beta}'
r1 = '${fparse (1 / dt + sqrt(disc)) / (2 * alpha)}'
r2 = '${fparse (1 / dt - sqrt(disc)) / (2 * alpha)}'
a1 = '${fparse (1 / dt + beta) / alpha}'
A = '${fparse (a1 - r2) / (r1 - r2)}'
B = '${fparse 1 - A}'

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
    # the explicit sources, from the fields at the start of the step
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
    # a_n at t = n dt, and p_n = s a_{n-1} / lambda (p_0 = s / lambda)
    [gold_c_compute]
      type = ParsedCompute
      buffer = gold_c
      expression = '(A * r1^(t / dt) + B * r2^(t / dt)) * sin(2*pi*x) * sin(2*pi*y)'
      constant_names = 'A B r1 r2 dt'
      constant_expressions = '${A} ${B} ${r1} ${r2} ${dt}'
      extra_symbols = true
    []
    [gold_phi_compute]
      type = ParsedCompute
      buffer = gold_phi
      expression = 'if(t < dt / 2, 1, A * r1^(t / dt - 1) + B * r2^(t / dt - 1)) * p
                    * sin(2*pi*x) * sin(2*pi*y)'
      constant_names = 'A B r1 r2 dt p'
      constant_expressions = '${A} ${B} ${r1} ${r2} ${dt} ${fparse s / lambda}'
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
    buffer = gold_c
    value_type = MAX
  []
  [max_phi]
    type = TensorExtremeValuePostprocessor
    buffer = gold_phi
    value_type = MAX
  []
  # zero exactly; the solver sets the mean of c the summed equations fix, whatever l_tol
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
