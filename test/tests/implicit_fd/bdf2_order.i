# Order of accuracy of variable-step BDF2 against the exact decay of a periodic Fourier mode,
# independent of the BDF2 weight formulas. u_t = D Laplacian(u) with u(0) = sin(2 pi x) sin(2 pi y)
# on the n x n periodic grid: the mode is an eigenvector of the discrete Laplacian with eigenvalue
# -lambda_h, so the time-exact solution is exp(-D lambda_h t) times the mode, and max_u is that
# amplitude times the largest grid value of the mode, sin(15 pi / 32)^2 for n = 32.
#
# The steps alternate H, 2H (ratios 2 and 1/2 after the backward Euler first step) to t = 0.12.
# error_scaled is |max_u - exact| / H^2 at t = 0.12. Measured: 51.1 at H = 0.01 and 40.9 at
# H = 0.005 (error ratio 5.0, second order). The Terminator stops the run with an error above 60,
# which a first-order scheme exceeds at both H: backward Euler gives 169 and 310, BDF2 with the
# step ratio transposed (dt_old <-> dt) 121 and 279.

D = 0.5
n = 32
H = 0.01
t_end = 0.12

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
  [lap_u]
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
      order = 2
    []
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = '${fparse -D}'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  root_compute = dudt
  l_tol = 1e-12
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
  [error_scaled]
    type = ParsedPostprocessor
    pp_names = max_u
    # lambda_h of the 3-point Laplacian, 2 (2 - 2 cos(2 pi h)) / h^2; zero before t_end
    constant_names = 'D H t_end lambda_h mode_max'
    constant_expressions = '${D} ${H} ${t_end} ${fparse 4*(1-cos(2*pi/n))*n^2} ${fparse sin(15*pi/32)^2}'
    expression = 'if(t > t_end - H / 2, abs(max_u - mode_max * exp(-D * lambda_h * t)) / H^2, 0)'
    use_t = true
  []
[]

[UserObjects]
  [second_order]
    type = Terminator
    expression = 'error_scaled > 60'
    error_level = ERROR
    message = 'The BDF2 error at t_end exceeds 60 H^2: the scheme is not second order'
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  end_time = ${t_end}
  [TimeStepper]
    type = TimeSequenceStepper
    # steps H, 2H, H, 2H, ...
    time_sequence = '0.01 0.03 0.04 0.06 0.07 0.09 0.1 0.12'
  []
[]

[Outputs]
  csv = true
[]
