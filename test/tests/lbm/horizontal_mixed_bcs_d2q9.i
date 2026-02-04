[Domain]
  dim = 2
  nx = 10
  ny = 10
  mesh_mode = DUMMY
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [f]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [f_bounce_back]
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
      constants = '0.0001 0.0005'
    []
    [initial_f]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
    [initial_f_bb]
      type = LBMEquilibrium
      buffer = f_bounce_back
      bulk = density
      velocity = velocity
    []
  []
  [Solve]
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
  []
  [Boundary]
    [left]
      type = LBMFixedZerothOrderBC
      buffer = f
      f = f
      value = 1.1
      boundary = left
    []
    [right]
      type = LBMFixedFirstOrderBC
      buffer = f
      f = f
      value = 0.0001
      boundary = right
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = f_bounce_back
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
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
  num_steps = 5
[]

[Outputs]
  file_base = horizontal_mixed_bcs_d2q9
  csv = true
[]
