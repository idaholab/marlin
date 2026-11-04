#
# The Kim-Kim-Suzuki model of test/tests/kks/KKS_no_flux_bc.i with the implicit finite-difference
# solver. The spectral input keeps no flux through x = -20, 20 with a smooth-boundary mask in a
# periodic box; here the domain is the strip -20 < x < 20 (periodic in y) with zero-flux faces.
#
#   dc/dt = div(M grad mu),  mu = dF/dc,
#   deta/dt = L kappa Laplacian(eta) - L dF/deta (at fixed c),
#
# F = h rho^2 (c - (1-h)(c_b - c_a) - c_a)^2 + (1-h) rho^2 (c + h (c_b - c_a) - c_b)^2
#     + w eta^2 (1-eta)^2,  h = eta^3 (6 eta^2 - 15 eta + 10).
#
# Both parabolas have the curvature 2 rho^2 in c, so d2F/dc2 = 4 exactly and mu = 4 c + g(eta)
# with g = mu - 4 c a function of eta only. The c step takes M 4 Laplacian(c) implicitly and
# M Laplacian(g) as a source; the eta step takes L kappa Laplacian(eta) implicitly and
# -L dF/deta as a source. The nonlinear solve iterates both sources to the fully implicit
# backward Euler step of the coupled system.
#
# No flux: the flux of c through a face is M (4 dc/dn + dg/dn); the faces split it as dc/dn = 0
# and dg/dn = 0 (Neumann faces of c and g), so the summed c equation conserves the total of c to
# round-off. eta has zero-flux faces as well.
#
# Output: total c, total eta and the free energy integral of F + kappa/2 |grad eta|^2, which
# decreases in every step here. The gold is the mirrored initial condition on the doubled periodic
# domain -20 < x < 60 (integrals halved), which the zero-flux faces reproduce to round-off.
#

# Constants for Initial Conditions
r = 30
l = 4.2

# Initial condition function for order parameter
eta_IC = '0.5*(1-tanh(2*(sqrt(x^2+y^2)-${r})/${l}))'

# Phase-field model parameters
kappa_eta = 5
rho_sq = 2
w = 1
M = 5
L = 5
c0_a = 0.3
c0_b = 0.7

# Expressions for switching function and bulk Gibbs energy
h_eta = 'eta^3*(6*eta^2-15*eta+10)'
F = '${h_eta}*(${rho_sq}*((c - (1-${h_eta})*(${c0_b} - ${c0_a}))-${c0_a})^2) + (1-${h_eta})*(${rho_sq}*((c + (${h_eta})*(${c0_b} - ${c0_a}))-${c0_b})^2 ) + ${w}*(eta^2)*(1-eta)^2'

# grid spacing 1 (about four cells across the interface width l)
nx = 40
ny = 100
dt = 0.1
num_steps = 10

[Domain]
  dim = 2
  nx = ${nx}
  ny = ${ny}
  xmin = -20
  xmax = 20
  ymin = -50
  ymax = 50
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
[]

[TensorBuffers]
  [c]
  []
  [eta]
  []
  [dcdt]
  []
  [detadt]
  []
  [lap_c]
  []
  [lap_eta]
  []
  [mu]
  []
  [g]
  []
  [rhs_c]
  []
  [rhs_eta]
  []
  # postprocessing
  [lap_eta_pp]
  []
  [energy]
  []
[]

[TensorComputes]
  [Initialize]
    [c_IC]
      type = ParsedCompute
      buffer = c
      expression = '0.6 + (${c0_a}-0.6)*${eta_IC}'
      extra_symbols = true
    []
    [eta_IC]
      type = ParsedCompute
      buffer = eta
      expression = '${eta_IC}'
      extra_symbols = true
    []
  []

  [BCs]
    [c_left]
      type = TensorNeumannBC
      variable = c
      boundary = left
    []
    [c_right]
      type = TensorNeumannBC
      variable = c
      boundary = right
    []
    [g_left]
      type = TensorNeumannBC
      variable = g
      boundary = left
    []
    [g_right]
      type = TensorNeumannBC
      variable = g
      boundary = right
    []
    [eta_left]
      type = TensorNeumannBC
      variable = eta
      boundary = left
    []
    [eta_right]
      type = TensorNeumannBC
      variable = eta
      boundary = right
    []
  []

  [Solve]
    # implicit operators
    [dcdt_op]
      type = TimeDerivativeOperator
      buffer = dcdt
      input = c
    []
    [lap_c_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_c
      input = c
      factor = '${fparse -M * 2 * rho_sq}'
    []
    [detadt_op]
      type = TimeDerivativeOperator
      buffer = detadt
      input = eta
    []
    [lap_eta_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_eta
      input = eta
      factor = '${fparse -L * kappa_eta}'
    []

    # sources, from the current iterate
    [explicit]
      [mu_compute]
        type = ParsedCompute
        buffer = mu
        expression = '${F}'
        inputs = 'c eta'
        derivatives = c
      []
      [g_compute]
        type = ParsedCompute
        buffer = g
        expression = 'mu - ${fparse 2 * rho_sq} * c'
        inputs = 'mu c'
      []
      [rhs_c_compute]
        type = FiniteDifferenceLaplacian
        buffer = rhs_c
        input = g
        factor = ${M}
      []
      [dF_deta]
        type = ParsedCompute
        buffer = rhs_eta
        expression = '-${L} * (${F})'
        inputs = 'c eta'
        derivatives = eta
      []
    []
  []

  [Postprocess]
    # with zero-flux faces sum |grad eta|^2 = -sum eta Laplacian(eta) (summation by parts)
    [lap_eta_pp]
      type = FiniteDifferenceLaplacian
      buffer = lap_eta_pp
      input = eta
    []
    [energy]
      type = ParsedCompute
      buffer = energy
      expression = '${F} - ${kappa_eta} / 2 * eta * lap_eta_pp'
      inputs = 'c eta lap_eta_pp'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = 'c eta'
  operators = 'dcdt_op lap_c_op; detadt_op lap_eta_op'
  source = 'rhs_c rhs_eta'
  root_compute = explicit
  l_tol = 1e-12
  nonlinear_solve = true
  nl_rel_tol = 1e-10
  nl_max_its = 200
  petsc_options_iname = '-snes_type -npc_snes_type'
  petsc_options_value = 'anderson newtonls'
[]

[Postprocessors]
  [total_c]
    type = TensorIntegralPostprocessor
    buffer = c
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_c_change]
    type = ChangeOverTimePostprocessor
    postprocessor = total_c
    change_with_respect_to_initial = true
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_eta]
    type = TensorIntegralPostprocessor
    buffer = eta
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [free_energy]
    type = TensorIntegralPostprocessor
    buffer = energy
    execute_on = 'INITIAL TIMESTEP_END'
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
  execute_on = 'INITIAL TIMESTEP_END'
[]
