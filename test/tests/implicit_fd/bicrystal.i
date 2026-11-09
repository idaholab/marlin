# The shrinking circular grain of PFHub benchmark 9a (the model of idaholab/marlin#173, scaled
# down): a circular grain eta1 of radius r0 inside a grain eta2, with
#
#   eta_i,t = L kappa Laplacian(eta_i) - L dF/deta_i,
#   F = m (eta1^4/4 - eta1^2/2 + eta2^4/4 - eta2^2/2 + 3/2 eta1^2 eta2^2 + 1/4),
#
# on a periodic square. Both fields are solved by one ImplicitFDSolver with the nonlinear solve,
# so a step is the fully implicit backward Euler step of both fields. In the sharp-interface limit
# the grain area falls at the constant rate dA/dt = -2 pi L kappa (L kappa = M gamma).
#
# The plain Picard iteration converges only for dt L f'' < 1 roughly (f'' = 2 m in the grains), as
# the field solves do not see the reaction; Anderson mixing (the tests' petsc_options
# '-snes_type -npc_snes_type' = 'anderson newtonls') extends that several times.
#
# The parameters here are small for a fast regression test, compared with the NumPy Newton
# reference bicrystal_newton.py. The area rate needs a grain much larger than the interface width
# (r0 = 100, n = 64, h = 5 on the command line).

# units: nm, microseconds, eV
L = 0.8545 # nm^3 / (eV us)
kappa = 210.6479 # eV / nm
m = 1.8724 # eV / nm^3
interface_width = 30
r0 = 25
n = 24
h = 4
dt = 0.5

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmin = '${fparse -n * h / 2}'
  xmax = '${fparse n * h / 2}'
  ymin = '${fparse -n * h / 2}'
  ymax = '${fparse n * h / 2}'
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [eta1]
  []
  [eta2]
  []
  [deta1]
  []
  [deta2]
  []
  [lap_eta1]
  []
  [lap_eta2]
  []
  [dF1]
  []
  [dF2]
  []
  [src1]
  []
  [src2]
  []
[]

[TensorComputes]
  [Initialize]
    [eta1_ic]
      type = ParsedCompute
      buffer = eta1
      expression = 'radius:=sqrt(x^2 + y^2); 0.5 * (1 - tanh(2 * (radius - ${r0}) / ${interface_width}))'
      extra_symbols = true
    []
    [eta2_ic]
      type = ParsedCompute
      buffer = eta2
      expression = 'radius:=sqrt(x^2 + y^2); 0.5 * (1 + tanh(2 * (radius - ${r0}) / ${interface_width}))'
      extra_symbols = true
    []
  []
  [Solve]
    [dF1_compute]
      type = ParsedCompute
      buffer = dF1
      expression = 'm * (eta1^4/4 - eta1^2/2 + eta2^4/4 - eta2^2/2 + 1.5*eta1^2*eta2^2 + 1/4)'
      constant_names = m
      constant_expressions = ${m}
      inputs = 'eta1 eta2'
      derivatives = eta1
    []
    [dF2_compute]
      type = ParsedCompute
      buffer = dF2
      expression = 'm * (eta1^4/4 - eta1^2/2 + eta2^4/4 - eta2^2/2 + 1.5*eta1^2*eta2^2 + 1/4)'
      constant_names = m
      constant_expressions = ${m}
      inputs = 'eta1 eta2'
      derivatives = eta2
    []
    [src1_compute]
      type = ParsedCompute
      buffer = src1
      expression = '-${L} * dF1'
      inputs = dF1
    []
    [src2_compute]
      type = ParsedCompute
      buffer = src2
      expression = '-${L} * dF2'
      inputs = dF2
    []
    [deta1_op]
      type = TimeDerivativeOperator
      buffer = deta1
      input = eta1
    []
    [deta2_op]
      type = TimeDerivativeOperator
      buffer = deta2
      input = eta2
    []
    [lap_eta1_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_eta1
      input = eta1
      factor = '${fparse -L * kappa}'
    []
    [lap_eta2_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_eta2
      input = eta2
      factor = '${fparse -L * kappa}'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = 'eta1 eta2'
  operators = 'deta1_op lap_eta1_op; deta2_op lap_eta2_op'
  source = 'src1 src2'
  l_tol = 1e-13
  nonlinear_solve = true
  nl_rel_tol = 1e-13
  nl_max_its = 200
[]

[Postprocessors]
  # the grain area (PFHub 9a's int_eta1)
  [area]
    type = TensorIntegralPostprocessor
    buffer = eta1
  []
  [max_eta1]
    type = TensorExtremeValuePostprocessor
    buffer = eta1
    value_type = MAX
  []
  [min_eta2]
    type = TensorExtremeValuePostprocessor
    buffer = eta2
    value_type = MIN
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 5
  dt = ${dt}
[]

[Outputs]
  csv = true
[]
