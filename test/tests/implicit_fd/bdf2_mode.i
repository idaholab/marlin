# Variable-step BDF2 diffusion u_t = D Laplacian(u) of a single periodic Fourier mode, the BDF2
# counterpart of be_mode.i. The mode is an eigenvector of the discrete Laplacian with eigenvalue
# -lambda_h, so its amplitude a follows the scalar BDF2 recurrence
#   (w0 a^{n+1} + w1 a^n + w2 a^{n-1}) / dt + D lambda_h a^{n+1} = 0,
# with the variable-step weights of the step (sum = dt + dt_old)
#   w0 = 1 + dt / sum, w1 = -sum / dt_old, w2 = dt^2 / (dt_old sum),
# and a backward Euler first step. bdf2_mode.py evaluates it for the step sequence below, whose
# step ratios dt/dt_old range from 0.5 to 2 (exercising the weights and the operator rebuild),
# and writes the gold of max_u; the solve must reproduce it up to the linear solver tolerance.

D = 0.5
n = 32
stencil = 3

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
      stencil_width = ${stencil}
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  root_compute = dudt
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 8
  [TimeStepper]
    type = TimeSequenceStepper
    # steps 0.01, 0.01, 0.02, 0.01, 0.015, 0.0075, 0.015, 0.03
    time_sequence = '0.01 0.02 0.04 0.05 0.065 0.0725 0.0875 0.1175'
  []
[]

[Outputs]
  csv = true
[]
