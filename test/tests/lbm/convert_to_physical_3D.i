dx = 0.01
dt = 0.1

nu_phys = 1.5e-5
rho_phys = 1.225

[Domain]
  dim = 3
  nx = 10
  ny = 10
  nz = 10
  xmax = '${fparse 10 * dx}'
  ymax = '${fparse 10 * dx}'
  zmax = '${fparse 10 * dx}'
  mesh_mode = DUMMY
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y Z'
[]

[Stencil]
  [d3q19]
    type = LBMD3Q19
  []
[]

[TensorBuffers]
  [f]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [feq]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [fpc]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [density]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [speed]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [physical_velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [physical_pressure]
    type = LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [initial_density]
      type = LBMConstantTensor
      buffer = density
      constants = 1.0
    []
    [initial_velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0 0.0 0.0'
    []
    [initial_equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [initial_distribution]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
    [initial_distribution_pc]
      type = LBMEquilibrium
      buffer = fpc
      bulk = density
      velocity = velocity
    []
  []
  [Solve]
    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [collision]
      type = LBMBGKCollision
      buffer = fpc
      f = f
      feq = feq
      tau0 = 'tau'
    []
    [density]
      type = LBMComputeDensity
      buffer = density
      f = f
    []
    [velocity]
      type = LBMComputeVelocity
      buffer = velocity
      f = f
      rho = density
      add_body_force = true
      body_force_x = 0.0001
    []
    [speed]
      type = LBMComputeVelocityMagnitude
      buffer = speed
      velocity = velocity
    []
    [residual]
      type = LBMComputeResidual
      buffer = speed
      speed = speed
    []
  []
  [Boundary]
    [top]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = top
    []
    [bottom]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = bottom
    []
  []
  [Postprocess]
    [physical_velocity]
      type = LBMComputePhysicalVelocity
      buffer = physical_velocity
      velocity = velocity
      tau = 'tau'
      nu = '${nu_phys}'
    []
    [physical_pressure]
      type = LBMComputePhysicalPressure
      buffer = physical_pressure
      rho = density
      rho0 = 1.0
      rho0_phys = '${rho_phys}'
      tau = 'tau'
      nu = '${nu_phys}'
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = fpc
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 10
  scalar_constant_names = 'tau'
  scalar_constant_values = '${fparse nu_phys * 3.0 * dt / (dx * dx) + 0.5}'
[]

[Postprocessors]
  [velocity_max]
    type = TensorExtremeValuePostprocessor
    buffer = velocity
    value_type = MAX
    outputs = none
  []
  [physical_velocity_max]
    type = TensorExtremeValuePostprocessor
    buffer = physical_velocity
    value_type = MAX
  []
  [physical_velocity_max_expected]
    type = ParsedPostprocessor
    expression = 'velocity_max * ${fparse dx / dt}'
    pp_names = 'velocity_max'
  []

  [density_max]
    type = TensorExtremeValuePostprocessor
    buffer = density
    value_type = MIN
    outputs = none
  []
  [physical_pressure_max]
    type = TensorExtremeValuePostprocessor
    buffer = physical_pressure
    value_type = MIN
  []
  [physical_pressure_max_expected]
    type = ParsedPostprocessor
    expression = '${rho_phys} / 3.0 * ${fparse (dx / dt)^2} * (density_max - 1.0)'
    pp_names = 'density_max'
  []
[]

[Executioner]
  type = Transient
  num_steps = 11
[]

[Outputs]
  csv = true
[]
