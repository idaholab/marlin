# Implicit solve of u_t - D Laplacian(u) = f on the unit square with boundary faces, against the
# manufactured solution
#
#   u = g + tf t q,  g = cos(3 pi x / 2) cos(pi y) + x y,  q = x^2 - y^2,
#
# with tf = 0 for the steady problem -D Laplacian(u) = f and tf = 1 for backward Euler.
# q is harmonic, so f = tf q - D Laplacian(g) is independent of time and the backward Euler
# step makes no time error: the error is the spatial one, provided the boundary values are
# taken at the time of the unknown, t + dt (at t it would be of order dt).
#
# Each face has a different condition and value source:
#   left   (x = 0)  Dirichlet   u = cos(pi y) - tf t y^2                     function
#   right  (x = 1)  Robin       ra u + rb du/dx = c                         function
#                                                      or buffer (right_buffer, steady)
#   bottom (y = 0)  Neumann     du/dn = -du/dy = -x                          buffer
#   top    (y = 1)  Neumann     du/dn = du/dy = x - 2 tf t                   function
#                                                      or buffer (top_buffer, values at t)

D = 0.5
n = 32
stencil = 3
tf = 0
ops = 'lap'
num_steps = 1
dt = 0.05
# Robin coefficients on the right face (the 5-point stencil supports ra = 1, rb = 0 only)
ra = 1
rb = 1

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
    expression = 'cos(1.5*pi*x) * cos(pi*y) + x*y + ${tf} * t * (x^2 - y^2)'
  []
  [f_func]
    type = ParsedFunction
    # Laplacian(g) = -13/4 pi^2 cos(3 pi x / 2) cos(pi y)
    expression = '${tf} * (x^2 - y^2) + ${D} * 3.25 * pi^2 * cos(1.5*pi*x) * cos(pi*y)'
  []
  [right_c]
    type = ParsedFunction
    # on x = 1: u = y + tf t (1 - y^2), du/dx = 3/2 pi cos(pi y) + y + 2 tf t
    expression = '${ra} * (y + ${tf} * t * (1 - y^2)) + ${rb} * (1.5*pi*cos(pi*y) + y + 2 * ${tf} * t)'
  []
  [bottom_flux]
    type = ParsedFunction
    expression = '-x'
  []
  [top_flux]
    type = ParsedFunction
    expression = 'x - 2 * ${tf} * t'
  []
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [lap_u]
  []
  [f]
  []
  [bottom_flux]
  []
  [right_c]
  []
  [top_flux]
  []
  [exact]
  []
  [diff]
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
    # a face value: depends on the tangential coordinate x only
    [bottom_flux]
      type = MooseFunctionTensor
      buffer = bottom_flux
      function = bottom_flux
    []
    [right_c]
      type = MooseFunctionTensor
      buffer = right_c
      function = right_c
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
      type = TensorRobinBC
      variable = u
      boundary = right
      a = ${ra}
      b = ${rb}
      c_function = right_c
    []
    # the Robin c from a buffer instead (active = 'left right_buffer bottom top'), steady only
    [right_buffer]
      type = TensorRobinBC
      variable = u
      boundary = right
      a = ${ra}
      b = ${rb}
      c_buffer = right_c
    []
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux_buffer = bottom_flux
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = top_flux
    []
    # the top flux from a buffer instead (active = 'left right bottom top_buffer'); a buffer
    # value is the one its compute wrote at the start of the substep, at t
    [top_buffer]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_buffer = top_flux
    []
  []
  [Solve]
    [top_flux]
      type = MooseFunctionTensor
      buffer = top_flux
      function = top_flux
    []
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = '${fparse -D}'
      stencil_width = ${stencil}
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
  num_steps = ${num_steps}
  dt = ${dt}
[]

[Outputs]
  csv = true
[]
