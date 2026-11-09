# Implicit solve of u_t + M Laplacian(Laplacian(u)) = f on the unit square with boundary faces,
# against the manufactured solution
#
#   u = g + s(t) h,  s = 1 + tf t,  h harmonic (Laplacian(h) = 0),
#
# so Laplacian(u) = Laplacian(g) and f = tf h + M Laplacian^2(g) does not depend on time. The
# biharmonic's face pairs are (u = value, Laplacian(u) = 0) on a Dirichlet face and
# (du/dn = flux, dLaplacian(u)/dn = 0) on a Neumann face, which u satisfies:
#   dirichlet:        g = sin(pi x) sin(pi y), h = x y; u = s x y on the faces
#   neumann:          g = cos(pi x) cos(pi y), h = x^2 - y^2; du/dn = 2 s on the right face,
#                     -2 s on the top face, 0 on the left and bottom faces
#   dirichlet_curved: g as dirichlet, h = x^2 - y^2; u = s h on the faces
#   neumann_curved:   g as neumann, h = x^3 - 3 x y^2; du/dn = 3 s y^2 (left), 3 s (1 - y^2)
#                     (right), 0 (bottom), -6 s x (top)
# The curved cases' face data are not linear along the faces, where d2u/dn2 (Dirichlet) or
# d3u/dn3 (Neumann) does not vanish, so they check Laplacian(u) = 0 and dLaplacian(u)/dn = 0
# rather than those.
# Laplacian(g) = -2 pi^2 g vanishes on the faces (dirichlet), its normal derivative does
# (neumann). tf = 0 is steady (Dirichlet only: with Neumann faces the biharmonic alone is
# singular). With tf = 1 u is linear in t and the face data are taken at t + dt, so backward
# Euler makes no time error and the error is the spatial one. Laplacian^2(g) = 4 pi^4 g.

M = 1e-3
n = 32
faces = dirichlet
tf = 1
ops = 'dudt bih'
num_steps = 2
dt = 0.01
linear_solver = TORCH_PCG

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
    expression = 'g + (1 + ${tf} * t) * h'
    symbol_names = 'g h'
    symbol_values = 'g_${faces} h_${faces}'
  []
  [f_func]
    type = ParsedFunction
    expression = '${tf} * h + 4 * pi^4 * ${M} * g'
    symbol_names = 'g h'
    symbol_values = 'g_${faces} h_${faces}'
  []
  [g_dirichlet]
    type = ParsedFunction
    expression = 'sin(pi*x) * sin(pi*y)'
  []
  [h_dirichlet]
    type = ParsedFunction
    expression = 'x*y'
  []
  [dirichlet_value]
    type = ParsedFunction
    expression = '(1 + ${tf} * t) * h'
    symbol_names = h
    symbol_values = h_${faces}
  []
  [g_neumann]
    type = ParsedFunction
    expression = 'cos(pi*x) * cos(pi*y)'
  []
  [h_neumann]
    type = ParsedFunction
    expression = 'x^2 - y^2'
  []
  [g_dirichlet_curved]
    type = ParsedFunction
    expression = 'sin(pi*x) * sin(pi*y)'
  []
  [h_dirichlet_curved]
    type = ParsedFunction
    expression = 'x^2 - y^2'
  []
  [g_neumann_curved]
    type = ParsedFunction
    expression = 'cos(pi*x) * cos(pi*y)'
  []
  [h_neumann_curved]
    type = ParsedFunction
    expression = 'x^3 - 3*x*y^2'
  []
  # du/dn = -dh/dx (left), dh/dx (right), -dh/dy (bottom), dh/dy (top), times 1 + tf t
  [flux_left_neumann]
    type = ParsedFunction
    expression = '0'
  []
  [flux_right_neumann]
    type = ParsedFunction
    expression = '2'
  []
  [flux_bottom_neumann]
    type = ParsedFunction
    expression = '0'
  []
  [flux_top_neumann]
    type = ParsedFunction
    expression = '-2'
  []
  [flux_left_neumann_curved]
    type = ParsedFunction
    expression = '3*y^2'
  []
  [flux_right_neumann_curved]
    type = ParsedFunction
    expression = '3*(1 - y^2)'
  []
  [flux_bottom_neumann_curved]
    type = ParsedFunction
    expression = '0'
  []
  [flux_top_neumann_curved]
    type = ParsedFunction
    expression = '-6*x'
  []
  # the flux functions are built in the Dirichlet cases too, unused
  [flux_left_dirichlet]
    type = ParsedFunction
    expression = '0'
  []
  [flux_right_dirichlet]
    type = ParsedFunction
    expression = '0'
  []
  [flux_bottom_dirichlet]
    type = ParsedFunction
    expression = '0'
  []
  [flux_top_dirichlet]
    type = ParsedFunction
    expression = '0'
  []
  [flux_left_dirichlet_curved]
    type = ParsedFunction
    expression = '0'
  []
  [flux_right_dirichlet_curved]
    type = ParsedFunction
    expression = '0'
  []
  [flux_bottom_dirichlet_curved]
    type = ParsedFunction
    expression = '0'
  []
  [flux_top_dirichlet_curved]
    type = ParsedFunction
    expression = '0'
  []
  [flux_left]
    type = ParsedFunction
    expression = '(1 + ${tf} * t) * q'
    symbol_names = q
    symbol_values = flux_left_${faces}
  []
  [flux_right]
    type = ParsedFunction
    expression = '(1 + ${tf} * t) * q'
    symbol_names = q
    symbol_values = flux_right_${faces}
  []
  [flux_bottom]
    type = ParsedFunction
    expression = '(1 + ${tf} * t) * q'
    symbol_names = q
    symbol_values = flux_bottom_${faces}
  []
  [flux_top]
    type = ParsedFunction
    expression = '(1 + ${tf} * t) * q'
    symbol_names = q
    symbol_values = flux_top_${faces}
  []
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [bih_u]
  []
  [f]
  []
  [exact]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    # the steady case starts from u = 0 (u_ic = 'f_ic'), so its solve does not start converged
    active = 'u_ic f_ic'
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
  []
  [BCs]
    # the Neumann case sets active = 'neumann_left neumann_right neumann_bottom neumann_top'
    active = 'dirichlet_left dirichlet_right dirichlet_bottom dirichlet_top'
    [dirichlet_left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value_function = dirichlet_value
    []
    [dirichlet_right]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value_function = dirichlet_value
    []
    [dirichlet_bottom]
      type = TensorDirichletBC
      variable = u
      boundary = bottom
      value_function = dirichlet_value
    []
    [dirichlet_top]
      type = TensorDirichletBC
      variable = u
      boundary = top
      value_function = dirichlet_value
    []
    [neumann_left]
      type = TensorNeumannBC
      variable = u
      boundary = left
      flux_function = flux_left
    []
    [neumann_right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux_function = flux_right
    []
    [neumann_bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux_function = flux_bottom
    []
    [neumann_top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = flux_top
    []
  []
  [Solve]
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih_u
      input = u
      factor = ${M}
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
  linear_solver = ${linear_solver}
  l_tol = 1e-13
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
  [mass]
    type = TensorIntegralPostprocessor
    buffer = u
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
