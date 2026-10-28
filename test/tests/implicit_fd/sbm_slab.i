# Smoothed boundary diffusion psi u_t = div(psi D grad u) in 1D (a strip 4 cells wide in y,
# periodic, as REAL_SPACE needs dim >= 2): the slab 0.5 < x < 1.5 of the periodic [0, 2],
# bounded by psi walls of half-width l (the cosine ramp of
# test/tests/kks/KKS_no_flux_bc.i, from psi = 0 at x_wall - l to 1 at x_wall + l, 1/2 at the
# wall). The SBM equation imposes zero flux at the walls, so as l -> 0 it tends to the sharp
# solution u = 1 + cos(pi (x - 1/2)) exp(-D pi^2 t) of the slab with Neumann faces. error is the
# largest |u - u_sharp| over the cells with psi = 1 at t_end, relative to the amplitude
# exp(-D pi^2 t_end). The grid resolves the wall with a fixed number of cells (h = l/4); with
# BDF2 steps of 1e-4 the time error is below 1e-6 (at dt = 1e-3 it is ~4e-5), so error is the
# SBM error of the wall width. mass_psi, the integral of psi u, is conserved.
# Measured (2026-10-28, PETSc, l_tol = 1e-12): error falls at order ~2 in l,
#   l       0.08     0.04     0.02     0.01     0.005
#   error   3.7e-4   1.05e-4  2.5e-5   6.1e-6   1.8e-6
# (orders 1.8, 2.1, 2.0, 1.8; the last approaches the time error).

D = 1
l = 0.02
cells_per_l = 4
t_end = 0.1
dt = 1e-4

[Domain]
  dim = 2
  nx = '${fparse 2 * cells_per_l / l}'
  ny = 4
  xmax = 2
  ymax = '${fparse 4 * l / cells_per_l}'
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Functions]
  [psi_func]
    type = ParsedFunction
    expression = 'if(x < 0.5 - l, 0, if(x > 0.5 + l, 1, 0.5 - 0.5 * cos(pi * (x - (0.5 - l)) / 2 / l)))
                  * if(x < 1.5 - l, 1, if(x > 1.5 + l, 0, 0.5 + 0.5 * cos(pi * (x - (1.5 - l)) / 2 / l)))'
    symbol_names = 'l'
    symbol_values = '${l}'
  []
  [u_func]
    type = ParsedFunction
    expression = '1 + cos(pi * (x - 0.5))'
  []
  [sharp_func]
    type = ParsedFunction
    expression = '1 + cos(pi * (x - 0.5)) * exp(-${D} * pi^2 * t)'
  []
[]

[TensorBuffers]
  [u]
  []
  [psi]
  []
  [psi_D]
  []
  [psi_u]
  []
  [sharp]
  []
  [error_cells]
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
  []
  [Solve]
    [dudt]
      type = CoefficientTimeDerivativeOperator
      buffer = dudt_u
      input = u
      coefficient = psi
      order = 2
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
    [sharp_compute]
      type = MooseFunctionTensor
      buffer = sharp
      function = sharp_func
    []
    [error_compute]
      type = ParsedCompute
      buffer = error_cells
      expression = 'if(psi >= 1, abs(u - sharp), 0) / ${fparse exp(-D * pi^2 * t_end)}'
      inputs = 'psi u sharp'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt divgrad'
  root_compute = dudt
  linear_solver = PETSC
  l_tol = 1e-12
[]

[Postprocessors]
  [error]
    type = TensorExtremeValuePostprocessor
    buffer = error_cells
    value_type = MAX
  []
  [mass_psi]
    type = TensorIntegralPostprocessor
    buffer = psi_u
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  end_time = ${t_end}
  dt = ${dt}
[]

[Outputs]
  # the end state only (1000 steps)
  [csv]
    type = CSV
    execute_on = FINAL
  []
[]
