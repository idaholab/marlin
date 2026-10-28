#
# The Kim-Kim-Suzuki model of test/tests/kks/KKS_no_flux_bc.i, with its smoothed boundary walls
# (psi, no flux through x = -20, 20 at a 90 degree contact angle, in the periodic box
# -50 < x, y < 50), with the implicit finite-difference solver. The psi-multiplied SBM equations
# (Yu, Chen and Thornton, arXiv:1107.5341, Eq. 4 and 22 without boundary sources) are
#
#   psi dc/dt = div(psi M grad mu),  mu = dF/dc = 4 c + g(eta),
#   psi deta/dt = div(psi L kappa grad eta) - psi L dF/deta (at fixed c),
#
# with F and g as in kks_no_flux.i. The c step takes psi dc/dt - div(4 psi M grad c) implicitly
# and div(psi M grad g) as a source; the eta step takes psi deta/dt - div(psi L kappa grad eta)
# implicitly and -psi L dF/deta as a source. The face coefficients are arithmetic means, so the
# psi = 0 cells next to a wall carry its zero-flux condition and the cells beyond are inactive
# (held at their initial values). The nonlinear solve iterates the sources to the fully implicit
# step.
#
# Output: the integrals of psi c (conserved to round-off) and psi eta, and the free energy
# integral of psi (F + kappa/2 |grad eta|^2), which decreases in every step here.
#
# Comparison with the spectral SBM (2026-10-28): KKS_no_flux_bc.i on the same grid
# (Domain/nx=100 Domain/ny=100, AB3 with 1000 substeps per step, the same psi) gives, at t = 1,
# int psi eta = 2329.46 against 2328.73 here (dt = 0.01, time-converged to 0.1), a gap of 0.73 in a
# change of about 129. At dx = 0.5 the two give 2329.539 and 2329.434: the gap falls with dx, as
# both converge to the same SBM solution (int psi c agrees to 1e-8 relative, as both conserve it).
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

# grid spacing 1 (about four cells across the interface width l and the wall width 2 l)
n = 100
dt = 0.1
num_steps = 10
order = 2

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmin = -50
  xmax = 50
  ymin = -50
  ymax = 50
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Functions]
  # the wall profile of KKS_no_flux_bc.i: 0 to 1 over x_min -+ l, 1 to 0 over x_max -+ l
  [psi_func]
    type = ParsedFunction
    expression = 'if(x<x_min-${l},0,if(x>x_min+${l},1,0.5-0.5*cos(pi*(x-(x_min-${l}))/2/${l}) )) * if(x<x_max-${l},1,if(x>x_max+${l},0,0.5+0.5*cos(pi*(x-(x_max-${l}))/2/${l}) ))'
    symbol_names = 'x_min x_max'
    symbol_values = '-20 20'
  []
[]

[TensorBuffers]
  [c]
  []
  [eta]
  []
  [psi]
  []
  [psi_M]
  []
  [psi_M_c]
  []
  [psi_L_kappa]
  []
  [dcdt]
  []
  [detadt]
  []
  [divgrad_c]
  []
  [divgrad_eta]
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
  [psi_c]
  []
  [psi_eta]
  []
  [divgrad_eta_pp]
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
    [psi_IC]
      type = MooseFunctionTensor
      buffer = psi
      function = psi_func
    []
    [psi_M_IC]
      type = ParsedCompute
      buffer = psi_M
      expression = '${M} * psi'
      inputs = psi
    []
    [psi_M_c_IC]
      type = ParsedCompute
      buffer = psi_M_c
      expression = '${fparse M * 2 * rho_sq} * psi'
      inputs = psi
    []
    [psi_L_kappa_IC]
      type = ParsedCompute
      buffer = psi_L_kappa
      expression = '${fparse L * kappa_eta} * psi'
      inputs = psi
    []
  []

  [Solve]
    # implicit operators
    [dcdt_op]
      type = CoefficientTimeDerivativeOperator
      buffer = dcdt
      input = c
      coefficient = psi
      order = ${order}
    []
    [divgrad_c_op]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_c
      input = c
      coefficient = psi_M_c
      factor = -1
      face_mean = arithmetic
    []
    [detadt_op]
      type = CoefficientTimeDerivativeOperator
      buffer = detadt
      input = eta
      coefficient = psi
      order = ${order}
    []
    [divgrad_eta_op]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_eta
      input = eta
      coefficient = psi_L_kappa
      factor = -1
      face_mean = arithmetic
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
        type = FiniteDifferenceDivGrad
        buffer = rhs_c
        input = g
        coefficient = psi_M
        face_mean = arithmetic
      []
      [dF_deta]
        type = ParsedCompute
        buffer = rhs_eta
        expression = '-${L} * psi * (${F})'
        inputs = 'c eta psi'
        derivatives = eta
      []
    []
  []

  [Postprocess]
    [psi_c_compute]
      type = ParsedCompute
      buffer = psi_c
      expression = 'psi * c'
      inputs = 'psi c'
    []
    [psi_eta_compute]
      type = ParsedCompute
      buffer = psi_eta
      expression = 'psi * eta'
      inputs = 'psi eta'
    []
    # on the periodic grid sum psi_face |grad eta|^2 = -sum eta div(psi grad eta) (summation by
    # parts)
    [divgrad_eta_pp]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_eta_pp
      input = eta
      coefficient = psi
      face_mean = arithmetic
    []
    [energy]
      type = ParsedCompute
      buffer = energy
      expression = 'psi * (${F}) - ${kappa_eta} / 2 * eta * divgrad_eta_pp'
      inputs = 'c eta psi divgrad_eta_pp'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = 'c eta'
  operators = 'dcdt_op divgrad_c_op; detadt_op divgrad_eta_op'
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
  [total_psi_c]
    type = TensorIntegralPostprocessor
    buffer = psi_c
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_psi_c_change]
    type = ChangeOverTimePostprocessor
    postprocessor = total_psi_c
    change_with_respect_to_initial = true
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_psi_eta]
    type = TensorIntegralPostprocessor
    buffer = psi_eta
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
