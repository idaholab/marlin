#
# The Kim-Kim-Suzuki no-flux model of test/tests/implicit_fd/kks_no_flux.i (see there for the
# equations, the split and the faces) run to t = 100 with an adaptive time step: the step grows
# while the nonlinear solve takes fewer than 15 iterations and is cut when it takes more than 40
# or fails. CPU or CUDA: on MPS (single precision) the nonlinear residual stalls near 1e-5 of its
# initial value.
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
  # as the grain relaxes the initial residual of a step falls to 1e-2, and 1e-10 of it is below
  # the residual's round-off (1e-12): stop at 1e-9, 2e-11 of the first step's residual
  nl_abs_tol = 1e-9
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

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'eta c'
    enable_hdf5 = true
    transpose = false
  []
[]

[Executioner]
  type = Transient
  end_time = 100
  dtmax = 10
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 0.1
    min_iterations = 15
    max_iterations = 40
    growth_factor = 1.5
  []
[]

[Outputs]
  csv = true
  execute_on = 'INITIAL TIMESTEP_END'
[]
