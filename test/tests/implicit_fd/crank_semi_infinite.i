# Crank, "The Mathematics of Diffusion" (2nd ed.), Sec. 3.3(i), eq. (3.13), with the implicit
# solver: a semi-infinite medium (x > 0) with zero initial concentration and constant surface
# concentration C1 = 1 at x = 0 has the exact solution
#   C(x, t) = erfc(x / (2 sqrt(D t))).
# The same problem as real_space/semi_infinite_diffusion.i, which steps it explicitly.
#
# Pseudo-1D (ny = 1, Y periodic), D = 1. The domain extends 10 diffusion lengths at the
# comparison time t = 1, so the far Dirichlet u = 0 does not measurably affect the solution.
# Backward Euler with substeps of dt / substeps: the error is first order in the substep.

substeps = 100

[Domain]
  dim = 2
  nx = 100
  ny = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 10
  ymin = -0.5
  ymax = 0.5
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [lap_u]
  []
  [u_exact]
  []
  [diff]
  []
[]

[TensorComputes]
  # no [Initialize]: buffers start at zero, the C0 = 0 initial condition
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      # far boundary, approximating the semi-infinite medium's x -> infinity condition
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 0
    []
  []
  [Solve]
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = -1
    []
  []
  [Postprocess]
    [u_exact]
      type = ParsedCompute
      buffer = u_exact
      expression = 'erfc(x / (2 * sqrt(t))) + 0*y'
      extra_symbols = true
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - u_exact)'
      inputs = 'u u_exact'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  root_compute = dudt
  substeps = ${substeps}
  l_tol = 1e-10
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  dt = 1
  num_steps = 1
[]

[Outputs]
  csv = true
[]
