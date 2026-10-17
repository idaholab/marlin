# Zero-flux Neumann boundary condition: a pure diffusion problem on a domain that
# is non-periodic (TensorNeumannBC, flux = 0) on X and periodic on Y, starting
# from a non-uniform initial condition. Zero-flux everywhere (physical faces AND
# the periodic Y wraparound) means total mass integral(u) must be conserved to
# near machine precision for all time, even as the field itself diffuses toward
# its spatially uniform mean -- this is the roadmap's required zero-flux Neumann
# conservation check (PR1 plan, Sec. "Verification").
[Domain]
  dim = 2
  nx = 20
  ny = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 1
  ymin = -1
  ymax = 1
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # Non-uniform in both x and y, and not matching the zero-flux condition
      # pointwise, so diffusion actually moves mass around internally while total
      # mass stays fixed; this is a stronger check than a field already at its
      # own steady state.
      expression = '1 + sin(2 * pi * x) + 0.3 * cos(pi * y)'
      extra_symbols = true
    []
  []
  [BCs]
    [left]
      type = TensorNeumannBC
      variable = u
      boundary = left
      flux = 0
    []
    [right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux = 0
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
    []
  []
[]

[TensorSolver]
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 10
[]

[Postprocessors]
  [mass]
    type = TensorIntegralPostprocessor
    buffer = u
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # dx=0.05, dy=0.25; same CFL margin as dirichlet_linear.i/robin_steady_state.i.
  dt = 1e-2
  num_steps = 400
[]

[Outputs]
  [out]
    type = CSV
  []
[]
