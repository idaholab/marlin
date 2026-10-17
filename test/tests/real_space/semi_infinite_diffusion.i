# Verification against Crank, "The Mathematics of Diffusion" (2nd ed.), Sec. 3.3(i),
# eq. (3.13): a semi-infinite medium (x > 0) with uniform initial concentration C0 = 0
# and constant surface concentration C1 maintained at x = 0 has the exact solution
#   C(x, t) = C1 * erfc(x / (2*sqrt(D*t)))
# (special case of (3.13) with C0 = 0, using erfc = 1 - erf).
#
# Pseudo-1D domain: ny = 1 with Y periodic, so the only non-trivial direction is X.
# D = 1, so sqrt(D*t) = 1 at t = 1 (the comparison time). The domain extends from
# x = 0 to x = 10*sqrt(D*t) = 10 diffusion lengths, far enough that the true
# semi-infinite solution's value there (erfc(10) ~ 2e-45) is indistinguishable from
# zero -- the finite far boundary's Dirichlet condition (clamped to the x=0 value,
# i.e. zero disturbance from the IC) does not measurably affect the solution near
# x = 0 by the comparison time.
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

[TensorComputes]
  # No [Initialize] block: PlainTensorBuffer::init() already zero-initializes every
  # buffer (torch::zeros(...)) before any compute runs, which is exactly the C0 = 0
  # initial condition this test needs -- avoids ParsedCompute's "0 + 0*y"/"0*x"-style
  # algebraic-simplification-to-a-bare-scalar pitfall (see dirichlet_laplacian_check.i
  # for the companion, nonzero-uniform-value case of the same underlying limitation).
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      # Far boundary, approximating the semi-infinite medium's x -> infinity
      # condition; diffusion has not reached here by the comparison time.
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 0
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
      factor = 1
    []
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
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 250
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
  dt = 1
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
