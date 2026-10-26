# Implicit solve of u_t - div(D grad u) = f on the unit square with a variable coefficient, against
# the manufactured solution
#
#   u = g + tf t,  g = cos(3 pi x / 2) cos(pi y) + x y,  D = 1 + x/2 + (3/10) x cos(pi y),
#
# with tf = 0 for the steady problem -div(D grad u) = f and tf = 1 for backward Euler. The time
# part is constant in space, so f = tf - div(D grad g) is independent of time and the backward
# Euler step makes no time error (the boundary values are taken at t + dt).
#
# Dirichlet faces at x = 0, 1 (u from u_exact) and Neumann faces at y = 0, 1 (du/dn = -x, x).
# dD/dy = 0 on the Neumann faces, where the face coefficient is the boundary cell's D. With
# TensorComputes/BCs/active='left right_robin bottom top' the face x = 1 is the Robin face
# u + du/dn / 2 = c, c from u_exact (u = y and du/dn = 3 pi cos(pi y) / 2 + y there).
#
# D_jump = 1 replaces D after the first step by a coefficient jumping by 100 at x = 1/2 (with
# the source f of the smooth D, so u is no longer u_exact): the solver must reassemble its
# coarse levels from the new D. The Solve computes see the time at the start of the step.

n = 32
tf = 0
ops = 'divgrad'
num_steps = 1
dt = 0.05
face_mean = harmonic
D_jump = 0

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
[]

[Functions]
  [u_exact]
    type = ParsedFunction
    expression = 'cos(1.5*pi*x) * cos(pi*y) + x*y + ${tf} * t'
  []
  [D_func]
    type = ParsedFunction
    expression = '1 + x/2 + 0.3*x*cos(pi*y)'
  []
  [f_func]
    type = ParsedFunction
    # f = tf - div(D grad g)
    expression = '${tf} - (-3*pi*x*(x - pi*sin(pi*y)*cos(1.5*pi*x))*sin(pi*y)/10
                  + (2*y - 3*pi*sin(1.5*pi*x)*cos(pi*y))*(3*cos(pi*y) + 5)/20
                  - 13*pi^2*(3*x*cos(pi*y) + 5*x + 10)*cos(1.5*pi*x)*cos(pi*y)/40)'
  []
  [bottom_flux]
    type = ParsedFunction
    expression = '-x'
  []
  [top_flux]
    type = ParsedFunction
    expression = 'x'
  []
  [right_c]
    type = ParsedFunction
    expression = 'y + (1.5 * pi * cos(pi * y) + y) / 2 + ${tf} * t'
  []
  [D_step]
    type = ParsedFunction
    expression = 'if(${D_jump} & t > ${fparse dt / 2}, 1 + 99 * (x > 0.5), D)'
    symbol_names = D
    symbol_values = D_func
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = MooseFunctionTensor
      buffer = u
      function = u_exact
    []
    [f_ic]
      type = MooseFunctionTensor
      buffer = f
      function = f_func
    []
    [D_ic]
      type = MooseFunctionTensor
      buffer = D
      function = D_func
    []
  []
  [BCs]
    active = 'left right bottom top'
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value_function = u_exact
    []
    [right]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value_function = u_exact
    []
    [right_robin]
      type = TensorRobinBC
      variable = u
      boundary = right
      a = 1
      b = 0.5
      c_function = right_c
    []
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux_function = bottom_flux
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = top_flux
    []
  []
  [Solve]
    # D evaluated before every solve (the solver's explicit computes)
    [D_update]
      type = MooseFunctionTensor
      buffer = D
      function = D_step
    []
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [divgrad]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_u
      input = u
      coefficient = D
      factor = -1
      face_mean = ${face_mean}
    []
  []
  [Postprocess]
    [exact_compute]
      type = MooseFunctionTensor
      buffer = exact
      function = u_exact
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - exact)'
      inputs = 'u exact'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = '${ops}'
  source = f
  linear_solver = TORCH_PCG
  l_tol = 1e-13
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
  num_steps = ${num_steps}
  dt = ${dt}
[]

[Outputs]
  csv = true
[]
