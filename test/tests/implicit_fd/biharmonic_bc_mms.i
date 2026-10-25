# Implicit solve of u_t + M Laplacian(Laplacian(u)) = f on the unit square with boundary faces,
# against the manufactured solution
#
#   u = g + s(t) h,  s = 1 + tf t,  h harmonic (Laplacian(h) = 0),
#
# so Laplacian(u) = Laplacian(g) and f = tf h + M Laplacian^2(g) does not depend on time. The
# biharmonic's face pairs are (u = value, Laplacian(u) = 0) on a Dirichlet face and
# (du/dn = flux, dLaplacian(u)/dn = 0) on a Neumann face, which u satisfies:
#   dirichlet: g = sin(pi x) sin(pi y), h = x y; u = s x y on the faces
#   neumann:   g = cos(pi x) cos(pi y), h = x^2 - y^2; du/dn = 2 s on the right face, -2 s on
#              the top face, 0 on the left and bottom faces
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
    expression = '(1 + ${tf} * t) * x*y'
  []
  [neumann_flux]
    type = ParsedFunction
    # du/dn on the right face; the top face's is its negative
    expression = '2 * (1 + ${tf} * t)'
  []
  [neumann_flux_top]
    type = ParsedFunction
    expression = '-2 * (1 + ${tf} * t)'
  []
  [g_neumann]
    type = ParsedFunction
    expression = 'cos(pi*x) * cos(pi*y)'
  []
  [h_neumann]
    type = ParsedFunction
    expression = 'x^2 - y^2'
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
      flux = 0
    []
    [neumann_right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux_function = neumann_flux
    []
    [neumann_bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux = 0
    []
    [neumann_top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = neumann_flux_top
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
