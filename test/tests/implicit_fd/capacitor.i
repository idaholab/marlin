# Parallel-plate capacitor with two dielectric slabs, -div(eps0 eps_r grad phi) = 0 on [0, L]^2:
# phi = V on the plate at x = 0, phi = 0 on the plate at x = L, no flux through y = 0, L, so phi
# depends on x only. eps_r = eps1 in slab 1 (thickness d1 = L/4) and eps2 in slab 2, the
# interface on a cell face. The charge per area on a plate is
#
#   Q/A = eps0 V / (d1/eps1 + d2/eps2).
#
# phi is linear in each slab and the discrete solution is that phi: the harmonic face mean of
# eps gives the exact series flux across the interface, and the Dirichlet ghost the exact flux
# at a plate. So the errors left are the linear solver's.
#
# metal = 1 adds metal plates of thickness t = L/8 at both ends (eps_r = eps_metal), with the
# Dirichlet faces at their outer faces; the dielectric is [t, L - t] (d1 = L/4, d2 = L/2) and
# Q/A has the plates' 2 t / eps_metal in its denominator.
#
# Charge measures:
# - charge_left, charge_right: the flux through each Dirichlet face. psi is a copy of phi with
#   no-flux faces there, so div(eps grad psi) differs from div(eps grad phi) = 0 only in the
#   cells next to those faces, by the face flux / h. With metal plates the potential drop across
#   a plate is about 1e-10 V, so round-off in phi (1e-15 V) makes these about 1e-5 off.
# - charge_dielectric: the flux eps0 eps1 (phi_i - phi_(i+1)) / h through the face x_f in the
#   middle of slab 1, from the weights +-eps0 eps1 / (L h^2) on the two cells next to it. It
#   has no such round-off.
#   x_f = t + L/8 is a cell face only when 8 divides n; otherwise charge_dielectric is 0.

n = 32
L = 1e-3
V = 10
eps0 = 8.8541878128e-12
eps1 = 1
eps2 = 100
metal = 0
eps_metal = 1e9
h = '${fparse L / n}'
t = '${fparse metal * L / 8}'
d1 = '${fparse L / 4}'
d2 = '${fparse L - 2 * t - d1}'
x_f = '${fparse t + d1 / 2}'

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = ${L}
  ymax = ${L}
  parallel_mode = REAL_SPACE
[]

[Functions]
  [eps_func]
    type = ParsedFunction
    expression = '${eps0} * if(x < ${t} | x > ${L} - ${t}, ${eps_metal}, if(x < ${t} + ${d1}, ${eps1}, ${eps2}))'
  []
  [phi_exact]
    type = ParsedFunction
    # q = eps0 V / (2 t / eps_metal + d1 / eps1 + d2 / eps2); phi falls by q dx / (eps0 eps_r)
    symbol_names = 'q'
    symbol_values = '${fparse eps0 * V / (2 * t / eps_metal + d1 / eps1 + d2 / eps2)}'
    expression = 'xm := min(x, ${t});
                  x1 := min(max(x - ${t}, 0), ${d1});
                  x2 := min(max(x - ${t} - ${d1}, 0), ${d2});
                  xr := max(x - ${L} + ${t}, 0);
                  ${V} - q / ${eps0} * ((xm + xr) / ${eps_metal} + x1 / ${eps1} + x2 / ${eps2})'
  []
  [left_cell_func]
    type = ParsedFunction
    expression = 'if(x < ${h}, 1, 0) + 0 * y'
  []
  [right_cell_func]
    type = ParsedFunction
    expression = 'if(x > ${L} - ${h}, 1, 0) + 0 * y'
  []
  [face_weight_func]
    type = ParsedFunction
    expression = '${fparse eps0 * eps1 / (L * h * h)}
                  * (if(x > ${x_f} - ${h} & x < ${x_f}, 1, 0) - if(x > ${x_f} & x < ${x_f} + ${h}, 1, 0))
                  + 0 * y'
  []
[]

[TensorComputes]
  [Initialize]
    [phi_ic]
      type = MooseFunctionTensor
      buffer = phi
      function = 0
    []
    [eps_ic]
      type = MooseFunctionTensor
      buffer = eps
      function = eps_func
    []
    [left_cell_ic]
      type = MooseFunctionTensor
      buffer = left_cell
      function = left_cell_func
    []
    [right_cell_ic]
      type = MooseFunctionTensor
      buffer = right_cell
      function = right_cell_func
    []
    [face_weight_ic]
      type = MooseFunctionTensor
      buffer = face_weight
      function = face_weight_func
    []
  []
  [BCs]
    [plate_v]
      type = TensorDirichletBC
      variable = phi
      boundary = left
      value = ${V}
    []
    [plate_0]
      type = TensorDirichletBC
      variable = phi
      boundary = right
      value = 0
    []
    [bottom]
      type = TensorNeumannBC
      variable = phi
      boundary = bottom
    []
    [top]
      type = TensorNeumannBC
      variable = phi
      boundary = top
    []
    [psi_left]
      type = TensorNeumannBC
      variable = psi
      boundary = left
    []
    [psi_right]
      type = TensorNeumannBC
      variable = psi
      boundary = right
    []
    [psi_bottom]
      type = TensorNeumannBC
      variable = psi
      boundary = bottom
    []
    [psi_top]
      type = TensorNeumannBC
      variable = psi
      boundary = top
    []
  []
  [Solve]
    [divgrad]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_phi
      input = phi
      coefficient = eps
      factor = -1
    []
  []
  [Postprocess]
    [psi_compute]
      type = ParsedCompute
      buffer = psi
      expression = 'phi'
      inputs = 'phi'
    []
    [divgrad_psi]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_psi
      input = psi
      coefficient = eps
    []
    [left_flux]
      type = ParsedCompute
      buffer = left_flux
      expression = '-left_cell * divgrad_psi / ${L}'
      inputs = 'left_cell divgrad_psi'
    []
    [right_flux]
      type = ParsedCompute
      buffer = right_flux
      expression = 'right_cell * divgrad_psi / ${L}'
      inputs = 'right_cell divgrad_psi'
    []
    [face_flux]
      type = ParsedCompute
      buffer = face_flux
      expression = 'face_weight * phi'
      inputs = 'face_weight phi'
    []
    [exact_compute]
      type = MooseFunctionTensor
      buffer = exact
      function = phi_exact
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(phi - exact) / ${V}'
      inputs = 'phi exact'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = phi
  operators = 'divgrad'
  linear_solver = TORCH_PCG
  l_tol = 1e-13
  l_max_its = 2000
[]

[Postprocessors]
  [charge_left]
    type = TensorIntegralPostprocessor
    buffer = left_flux
  []
  [charge_right]
    type = TensorIntegralPostprocessor
    buffer = right_flux
  []
  [charge_dielectric]
    type = TensorIntegralPostprocessor
    buffer = face_flux
  []
  [charge_exact]
    type = ParsedPostprocessor
    expression = '${fparse eps0 * V / (2 * t / eps_metal + d1 / eps1 + d2 / eps2)}'
  []
  [charge_left_error]
    type = ParsedPostprocessor
    expression = 'abs(charge_left / charge_exact - 1)'
    pp_names = 'charge_left charge_exact'
  []
  [charge_right_error]
    type = ParsedPostprocessor
    expression = 'abs(charge_right / charge_exact - 1)'
    pp_names = 'charge_right charge_exact'
  []
  [charge_dielectric_error]
    type = ParsedPostprocessor
    expression = 'abs(charge_dielectric / charge_exact - 1)'
    pp_names = 'charge_dielectric charge_exact'
  []
  [phi_error]
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
  num_steps = 1
[]

[Outputs]
  csv = true
[]
