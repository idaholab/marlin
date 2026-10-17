# Verification against an independent analytic solution: the steady 1D elliptic
# Robin-BC benchmark from OpenGeoSys's documentation,
# https://www.opengeosys.org/6.5.7/docs/benchmarks/elliptic/elliptic-robin/
# ("second benchmark": Dirichlet at x=0, Robin at x=1). The governing equation is
# k*Laplacian(h) = 0 on [0,1] with k=1, boundary conditions
#   h(0) = g_D,                 dh/dn|_{x=1} = alpha*(h0 - h(1)),
# and exact linear solution
#   h(x) = [alpha*(h0 - g_D)/(1 + alpha)] * x + g_D.
# TensorRobinBC's convention is a*u + b*du/dn = c; the OGS Robin condition
# du/dn = alpha*(h0 - u) rearranges to alpha*u + du/dn = alpha*h0, i.e.
# a = alpha, b = 1, c = alpha*h0.
#
# Deliberately uses alpha = +2 (not OGS's own alpha = -2): TensorRobinBC's
# ghost-fill coefficient on u0 is -(a/2 - b/dx)/(a/2 + b/dx), which has
# magnitude < 1 (damping) for a > 0 but magnitude > 1 (amplifying) for a < 0 at
# any dx. OGS's alpha = -2 is solved there as a pure steady linear system, with
# no explicit time relaxation; here the IC is relaxed toward steady state via
# RealSpaceForwardEuler, which diverges under an amplifying ghost coefficient
# (confirmed by direct run: max_error grows to 1e6+ by t=4 with alpha = -2).
# alpha = +2 is the stable branch of the exact same closed-form linear solution
# above, physically a boundary that loses more heat the hotter it is (ordinary
# convective/radiative cooling), consistent with a stable relaxation. With
# alpha = 2, h0 = 1.5, g_D = 1, the exact solution is h(x) = x/3 + 1.
#
# Pseudo-1D domain: ny = 1 with Y periodic, so the only non-trivial direction is X.
# The IC deliberately does not match the steady profile (nor the BCs pointwise),
# so this is a genuine relaxation-to-steady-state check, not an instantaneous one
# (cf. dirichlet_linear.i, the equivalent check for the pure-Dirichlet case).
[Domain]
  dim = 2
  nx = 20
  ny = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 1
  ymin = -1
  ymax = 1
[]

[Functions]
  [u_ic_func]
    type = ParsedFunction
    expression = '0.5'
  []
  [u_exact_func]
    type = ParsedFunction
    expression = '1 + x / 3'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      # Not the steady solution (constant, no x-dependence) and not matching the BCs
      # pointwise.
      type = MooseFunctionTensor
      buffer = u
      function = u_ic_func
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      # alpha*u + du/dn = alpha*h0, with alpha = 2, h0 = 1.5.
      type = TensorRobinBC
      variable = u
      boundary = right
      a = 2
      b = 1
      c = 3
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
    []
    [u_exact]
      type = MooseFunctionTensor
      buffer = u_exact
      function = u_exact_func
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
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 10
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # dx=0.05, dy=0.25; explicit-diffusion CFL limit ~1/(2*(1/dx^2+1/dy^2)) ~= 1.2e-3,
  # substeps=10 above divides dt into 10 explicit sub-updates, so the effective
  # per-substep step is dt/10 and must itself respect the CFL limit (same grid and
  # margin as dirichlet_linear.i).
  dt = 1e-2
  num_steps = 400
[]

[Outputs]
  [out]
    type = CSV
  []
[]
