# Crank, "The Mathematics of Diffusion" (2nd ed.), Sec. 2.5, eq. (2.15), with the implicit
# solver: an instantaneous planar source C0 = 1 confined to the slab -h < x < h, h = 1,
# diffusing into an infinite medium, has the exact solution
#   C(x, t) = (C0/2) (erf((h - x) / (2 sqrt(D t))) + erf((h + x) / (2 sqrt(D t)))).
# The same problem as real_space/thin_slab_diffusion.i, which steps it explicitly.
#
# Pseudo-1D (ny = 1, Y periodic), D = 1. The X faces are 10 diffusion lengths beyond the slab
# edge at the comparison time t = 1, so the far Dirichlet u = 0 does not measurably affect the
# solution.
# Backward Euler with substeps of dt / substeps: the error is first order in the substep.

substeps = 100

[Domain]
  dim = 2
  # dx = 0.1 puts the slab edges x = +/-1 on cell faces, so the box IC holds the slab mass
  nx = 220
  ny = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = -11
  xmax = 11
  ymin = -0.5
  ymax = 0.5
[]

[Functions]
  [slab]
    type = ParsedFunction
    expression = 'if(abs(x) < 1, 1, 0)'
  []
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
  [Initialize]
    [u_ic]
      type = MooseFunctionTensor
      buffer = u
      function = slab
    []
  []
  [BCs]
    # far boundaries, approximating the infinite medium
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 0
    []
    [right]
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
      expression = '0.5 * (erf((1 - x) / (2 * sqrt(t))) + erf((1 + x) / (2 * sqrt(t)))) + 0*y'
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
