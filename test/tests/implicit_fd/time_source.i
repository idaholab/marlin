# u_t = t from u = 0: the time level of the sources. Backward Euler with the source at t^{n+1}
# (the nonlinear solve) gives u^n = dt^2 n (n + 1) / 2 = t (t + dt) / 2, the lagged scheme with the
# source at t^n gives t (t - dt) / 2. The postprocessor error is the difference from the first.

dt = 0.1
n = 4

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Functions]
  [f_func]
    type = ParsedFunction
    expression = 't'
  []
  [gold_func]
    type = ParsedFunction
    expression = 't * (t + ${dt}) / 2'
  []
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [f]
  []
  [gold]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ConstantTensor
      buffer = u
      real = 0
    []
  []
  [Solve]
    [f_compute]
      type = MooseFunctionTensor
      buffer = f
      function = f_func
    []
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
  []
  [Postprocess]
    [gold_compute]
      type = MooseFunctionTensor
      buffer = gold
      function = gold_func
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - gold)'
      inputs = 'u gold'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = dudt
  source = f
  linear_solver = TORCH_PCG
  l_tol = 1e-13
  nonlinear_solve = true
[]

[Postprocessors]
  [error]
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
  num_steps = 5
  dt = ${dt}
[]

[Outputs]
  csv = true
[]
