# Smoothed boundary diffusion psi u_t = div(psi D grad u) on a periodic square, with psi = 1 in
# the band |x - 1/2| < 0.2, a cosine ramp to 0 at |x - 1/2| = 0.3 and psi = 0 outside, on the
# step sequence of bdf2_mode.i. The face coefficient is the arithmetic mean of psi D, so:
# - the psi = 0 cells next to the ramp have a positive face coefficient towards it and no time
#   derivative weight: they are solved, and act as ghost cells of the no-flux condition;
# - the cells further out have a zero row and a zero right-hand side: they are inactive and keep
#   u^0. held_change is max |u - u^0| over them, which must be exactly 0.
# sbm_inactive.py solves the same steps on the active cells alone (scipy sparse direct), the gold
# of the integrals of u, psi u and psi u^2. The integral of psi u is conserved, and that of
# psi u^2 decays (the extremes of u are those of the held u^0).

D = 0.5
n = 32
order = 2
# the source of the error test: nonzero where psi = 0
source_expression = '0'

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Functions]
  [psi_func]
    type = ParsedFunction
    expression = 'r := abs(x - 0.5); if(r < 0.2, 1, if(r < 0.3, 0.5 * (1 + cos(pi * (r - 0.2) / 0.1)), 0))'
  []
  [u_func]
    type = ParsedFunction
    expression = '1 + 0.5 * sin(2 * pi * x) * cos(2 * pi * y) + 0.25 * cos(4 * pi * y)'
  []
  [f_func]
    type = ParsedFunction
    expression = '${source_expression}'
  []
  # the inactive cells: psi = 0 on the cell and on its neighbours, |x - 1/2| - h > 0.3 (the cell
  # centres are at |x - 1/2| = (k + 1/2) h, h = 1/32, so no centre is near the cut)
  [inactive_func]
    type = ParsedFunction
    expression = 'if(abs(x - 0.5) - 1 / ${n} > 0.3, 1, 0)'
  []
[]

[TensorBuffers]
  [u]
  []
  [u0]
  []
  [psi]
  []
  [psi_D]
  []
  [psi_u]
  []
  [psi_u2]
  []
  [inactive]
  []
  [held]
  []
  [f]
  []
  [dudt_u]
  []
  [divgrad_u]
  []
[]

[TensorComputes]
  [Initialize]
    [psi_ic]
      type = MooseFunctionTensor
      buffer = psi
      function = psi_func
    []
    [psi_D_ic]
      type = ParsedCompute
      buffer = psi_D
      expression = '${D} * psi'
      inputs = psi
    []
    [u_ic]
      type = MooseFunctionTensor
      buffer = u
      function = u_func
    []
    [u0_ic]
      type = ParsedCompute
      buffer = u0
      expression = 'u'
      inputs = u
    []
    [f_ic]
      type = MooseFunctionTensor
      buffer = f
      function = f_func
    []
    [inactive_ic]
      type = MooseFunctionTensor
      buffer = inactive
      function = inactive_func
    []
  []
  [Solve]
    [dudt]
      type = CoefficientTimeDerivativeOperator
      buffer = dudt_u
      input = u
      coefficient = psi
      order = ${order}
    []
    [divgrad]
      type = FiniteDifferenceDivGrad
      buffer = divgrad_u
      input = u
      coefficient = psi_D
      factor = -1
      face_mean = arithmetic
    []
  []
  [Postprocess]
    [psi_u_compute]
      type = ParsedCompute
      buffer = psi_u
      expression = 'psi * u'
      inputs = 'psi u'
    []
    [psi_u2_compute]
      type = ParsedCompute
      buffer = psi_u2
      expression = 'psi * u^2'
      inputs = 'psi u'
    []
    [held_compute]
      type = ParsedCompute
      buffer = held
      expression = 'inactive * abs(u - u0)'
      inputs = 'inactive u u0'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt divgrad'
  root_compute = dudt
  source = f
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
  [min_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
  []
  [mass]
    type = TensorIntegralPostprocessor
    buffer = u
  []
  [mass_psi]
    type = TensorIntegralPostprocessor
    buffer = psi_u
  []
  [energy_psi]
    type = TensorIntegralPostprocessor
    buffer = psi_u2
  []
  [held_change]
    type = TensorExtremeValuePostprocessor
    buffer = held
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 8
  [TimeStepper]
    type = TimeSequenceStepper
    # steps 0.01, 0.01, 0.02, 0.01, 0.015, 0.0075, 0.015, 0.03
    time_sequence = '0.01 0.02 0.04 0.05 0.065 0.0725 0.0875 0.1175'
  []
[]

[Outputs]
  csv = true
[]
