#
# PFHub benchmark 6a (spinodal decomposition with electrostatics, square domain) with the
# implicit finite-difference solver:
#
#   dc/dt = div(M grad mu),  mu = f'(c) - kappa Laplacian(c) + k phi,
#   Laplacian(phi) = -k c / epsilon,
#
# f = rho_s (c - c_alpha)^2 (c_beta - c)^2, M = 5, kappa = 2, rho_s = 5, c_alpha = 0.3,
# c_beta = 0.7, k = 0.09, epsilon = 90, on 0 <= x, y <= 100.
#
# Faces: no flux of c everywhere; phi = 0 at x = 0, phi = sin(y/7) at x = 100, dphi/dn = 0 at
# y = 0, 100.
#
# The fully implicit version of 6a_implicit_fd.i: one ImplicitFDSolver solves c (transient) and
# phi (steady), and the nonlinear solve iterates each step to the backward Euler step of both,
#
#   (c - c^n)/dt + M kappa Laplacian(Laplacian(c)) = M Laplacian(f'(c) + k phi),
#   -Laplacian(phi) = k c / epsilon,
#
# all at t^{n+1}. The biharmonic and the Laplacian of phi are the field solves; f'(c) and the
# coupling are re-evaluated at every iterate. Anderson mixing over the field solves (as its
# nonlinear preconditioner) converges at steps where the plain Picard iteration does not.
#
# The time step follows the nonlinear iterations: it grows by 1.2 after a step of fewer than 8,
# halves after more than 15, and is halved and retried on a failed solve. The iteration count
# says nothing of the time error of backward Euler, which dtmax bounds. The first step starts
# from the initial guess of phi below;
# the iteration replaces it by the steady solution of c^1, so no lag is left. The free energy at
# t = 0 is that of the guess.
#
# No flux: the total flux of c through a face is M (dB/dn - kappa dLaplacian(c)/dn) with
# B = f'(c) + k phi, and in this scheme only the total enters the update, so the faces split
# it as dB/dn = 0 (Neumann faces of B) and dc/dn = dLaplacian(c)/dn = 0 (the biharmonic's
# Neumann pair). Any split with a zero total gives the same c to round-off; the summed
# equations conserve the total of c, which the solver keeps to round-off.
#
# Outputs: the free energy F = integral of kappa/2 |grad c|^2 + f(c) + k c phi / 2 against time,
# and c and phi on the cuts x = 50 and y = 50 at t = 400 from the HDF5 output
# (cut_planes.py).
#

n = 200
L = 100
end_time = 400

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = ${L}
  ymax = ${L}
  parallel_mode = REAL_SPACE
[]

[TensorBuffers]
  [c]
  []
  [phi]
  []
  [dcdt]
  []
  [bih_c]
  []
  [lap_phi]
  []
  [B]
  []
  [rhs_c]
  []
  [rhs_phi]
  []
  # postprocessing
  [lap_c]
  []
  [F]
  []
[]

[Functions]
  [phi_right]
    type = ParsedFunction
    expression = 'sin(y / 7)'
  []
[]

[TensorComputes]
  [Initialize]
    [c_ic]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'c0 + c1 * (cos(0.2*x)*cos(0.11*y) + (cos(0.13*x)*cos(0.087*y))^2
                    + cos(0.025*x - 0.15*y)*cos(0.07*x - 0.02*y))'
      constant_names = 'c0 c1'
      constant_expressions = '0.5 0.04'
    []
    # the initial guess of phi (it meets the plate values); the first step replaces it by the
    # steady solution of c^0
    [phi_ic]
      type = ParsedCompute
      buffer = phi
      extra_symbols = true
      expression = 'x / ${L} * sin(y / 7)'
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
    [c_bottom]
      type = TensorNeumannBC
      variable = c
      boundary = bottom
    []
    [c_top]
      type = TensorNeumannBC
      variable = c
      boundary = top
    []
    [B_left]
      type = TensorNeumannBC
      variable = B
      boundary = left
    []
    [B_right]
      type = TensorNeumannBC
      variable = B
      boundary = right
    []
    [B_bottom]
      type = TensorNeumannBC
      variable = B
      boundary = bottom
    []
    [B_top]
      type = TensorNeumannBC
      variable = B
      boundary = top
    []
    [phi_left]
      type = TensorDirichletBC
      variable = phi
      boundary = left
      value = 0
    []
    [phi_right]
      type = TensorDirichletBC
      variable = phi
      boundary = right
      value_function = phi_right
    []
    [phi_bottom]
      type = TensorNeumannBC
      variable = phi
      boundary = bottom
    []
    [phi_top]
      type = TensorNeumannBC
      variable = phi
      boundary = top
    []
  []

  [Solve]
    # implicit operators
    [dcdt]
      type = TimeDerivativeOperator
      buffer = dcdt
      input = c
    []
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih_c
      input = c
      factor = 10 # M * kappa
    []
    [lap_phi_op]
      type = FiniteDifferenceLaplacian
      buffer = lap_phi
      input = phi
      factor = -1
    []

    # explicit sources, re-evaluated on every iterate of c and phi
    [explicit]
      [B_compute]
        type = ParsedCompute
        buffer = B
        expression = '2 * rho_s * (c - c_alpha) * (c_beta - c) * (c_alpha + c_beta - 2 * c) + k * phi'
        constant_names = 'rho_s c_alpha c_beta k'
        constant_expressions = '5     0.3     0.7    0.09'
        inputs = 'c phi'
      []
      [rhs_c_compute]
        type = FiniteDifferenceLaplacian
        buffer = rhs_c
        input = B
        factor = 5 # M
      []
      [rhs_phi_compute]
        type = ParsedCompute
        buffer = rhs_phi
        expression = 'k / epsilon * c'
        constant_names = 'k epsilon'
        constant_expressions = '0.09 90'
        inputs = c
      []
    []
  []

  [Postprocess]
    # With zero-flux ghosts sum |grad c|^2 = -sum c Laplacian(c) for the 3-point Laplacian
    # (summation by parts, the face differences on the boundary vanish), so the gradient
    # energy (kappa/2)|grad c|^2 integrates to the same total as -(kappa/2) c Laplacian(c).
    [lap_c]
      type = FiniteDifferenceLaplacian
      buffer = lap_c
      input = c
    []
    [F]
      type = ParsedCompute
      buffer = F
      expression = 'rho_s * (c-c_alpha)^2 * (c_beta-c)^2 - kappa / 2 * c * lap_c + k * c * phi / 2'
      constant_names = 'rho_s c_alpha c_beta kappa k'
      constant_expressions = '5     0.3     0.7    2     0.09'
      inputs = 'c lap_c phi'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = 'c phi'
  operators = 'dcdt bih; lap_phi_op'
  source = 'rhs_c rhs_phi'
  root_compute = explicit
  l_tol = 1e-10
  nonlinear_solve = true
  petsc_options_iname = '-snes_type -npc_snes_type'
  petsc_options_value = 'anderson newtonls'
[]

[Postprocessors]
  [free_energy]
    type = TensorIntegralPostprocessor
    buffer = F
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
  []
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
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  end_time = ${end_time}
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 1e-4
    # Anderson's iterations grow slowly with dt: a window of 8-15 takes about 4 times the step
    # of 6-12 for 1.4 times the iterations per step
    min_iterations = 8
    max_iterations = 15
    growth_factor = 1.2
    cutback_factor = 0.5
  []
  # bounds the first-order time error of backward Euler: at n = 100, F is off by at most 0.06 to
  # t = 50 against dtmax = 0.005; the iteration window alone settles near dt = 0.15 by t = 1,
  # with three times that
  dtmax = 0.05
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c phi'
    enable_hdf5 = true
  []
[]

[Outputs]
  csv = true
  execute_on = 'INITIAL TIMESTEP_END'
[]
