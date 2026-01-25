[Domain]
  dim = 3
  nx = 10
  ny = 10
  nz = 10
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
    type=LBMTensorBuffer
    buffer_type = mv
  []
  [density]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [speed]
    type=LBMTensorBuffer
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
      type=LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [collision]
      type=LBMBGKCollision
      buffer = fpc
      f = f
      feq = feq
      tau0 = 1.0
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
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = fpc
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 10
[]

[Postprocessors]
  [velocity_min]
    type = TensorExtremeValuePostprocessor
    buffer = velocity
    value_type = MIN
  []
  [velocity_max]
    type = TensorExtremeValuePostprocessor
    buffer = velocity
    value_type = MAX
  []
  [speed_min]
    type = TensorExtremeValuePostprocessor
    buffer = speed
    value_type = MIN
  []
  [speed_max]
    type = TensorExtremeValuePostprocessor
    buffer = speed
    value_type = MAX
  []
  [density_min]
    type = TensorExtremeValuePostprocessor
    buffer = density
    value_type = MIN
  []
  [densty_max]
    type = TensorExtremeValuePostprocessor
    buffer = density
    value_type = MAX
  []
[]

[Executioner]
  type = Transient
  num_steps = 11
[]

[Outputs]
  file_base = channel3D
  csv = true
[]

