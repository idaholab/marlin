[Domain]
  dim = 2
  nx = 11
  ny = 11
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
  [binary_media]
    type = LBMTensorBuffer
    file = two_fluid_media.h5
    is_integer = true
    buffer_type = ms
  []

  # 1 inside the region of each fluid, 0 elsewhere
  [is_1]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [is_2]
    type = LBMTensorBuffer
    buffer_type = ms
  []

  # fluid 1
  [f1]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [feq1]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [fpc1]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [rho1]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [rho1_safe]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [u1]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [speed1]
    type = LBMTensorBuffer
    buffer_type = ms
  []

  # fluid 2
  [f2]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [feq2]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [fpc2]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [rho2]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [rho2_safe]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [u2]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [speed2]
    type = LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [init_is_1]
      type = ParsedCompute
      buffer = is_1
      inputs = binary_media
      expression = 'if(binary_media == 1, 1.0, 0.0)'
    []
    [init_is_2]
      type = ParsedCompute
      buffer = is_2
      inputs = binary_media
      expression = 'if(binary_media == 2, 1.0, 0.0)'
    []

    # fluid 1
    # uniform everywhere; only solid_values confines fluid 1 to its region
    [init_rho1]
      type = LBMConstantTensor
      buffer = rho1
      constants = 1.0
    []
    [init_u1]
      type = LBMConstantTensor
      buffer = u1
      constants = '0.0 0.0'
    []
    [init_feq1]
      type = LBMEquilibrium
      buffer = feq1
      bulk = rho1
      velocity = u1
      solid_values = '0 2'
    []
    [init_f1]
      type = LBMEquilibrium
      buffer = f1
      bulk = rho1
      velocity = u1
      solid_values = '0 2'
    []
    [init_fpc1]
      type = LBMEquilibrium
      buffer = fpc1
      bulk = rho1
      velocity = u1
      solid_values = '0 2'
    []

    # fluid 2
    # uniform everywhere; only solid_values confines fluid 2 to its region
    [init_rho2]
      type = LBMConstantTensor
      buffer = rho2
      constants = 1.01
    []
    [init_u2]
      type = LBMConstantTensor
      buffer = u2
      constants = '0.0 0.0'
    []
    [init_feq2]
      type = LBMEquilibrium
      buffer = feq2
      bulk = rho2
      velocity = u2
      solid_values = '0 1'
    []
    [init_f2]
      type = LBMEquilibrium
      buffer = f2
      bulk = rho2
      velocity = u2
      solid_values = '0 1'
    []
    [init_fpc2]
      type = LBMEquilibrium
      buffer = fpc2
      bulk = rho2
      velocity = u2
      solid_values = '0 1'
    []
  []

  [Solve]
    # fluid 1
    [rho1]
      type = LBMComputeDensity
      buffer = rho1
      f = f1
      solid_values = '0 2'
    []
    # rho1 is zero outside its region; divide by 1 there instead of 0
    [rho1_safe]
      type = ParsedCompute
      buffer = rho1_safe
      inputs = 'rho1 is_1'
      expression = 'rho1 + 1.0 - is_1'
    []
    [u1]
      type = LBMComputeVelocity
      buffer = u1
      f = f1
      rho = rho1_safe
      add_body_force = true
      body_force_x = g1
      solid_values = '0 2'
    []
    [feq1]
      type = LBMEquilibrium
      buffer = feq1
      bulk = rho1
      velocity = u1
      solid_values = '0 2'
    []
    [collision1]
      type = LBMBGKCollision
      buffer = fpc1
      f = f1
      feq = feq1
      tau0 = tau
      solid_values = '0 2'
    []
    [speed1]
      type = LBMComputeVelocityMagnitude
      buffer = speed1
      velocity = u1
      solid_values = '0 2'
    []

    # fluid 2
    [rho2]
      type = LBMComputeDensity
      buffer = rho2
      f = f2
      solid_values = '0 1'
    []
    [rho2_safe]
      type = ParsedCompute
      buffer = rho2_safe
      inputs = 'rho2 is_2'
      expression = 'rho2 + 1.0 - is_2'
    []
    [u2]
      type = LBMComputeVelocity
      buffer = u2
      f = f2
      rho = rho2_safe
      add_body_force = true
      body_force_x = g2
      solid_values = '0 1'
    []
    [feq2]
      type = LBMEquilibrium
      buffer = feq2
      bulk = rho2
      velocity = u2
      solid_values = '0 1'
    []
    [collision2]
      type = LBMBGKCollision
      buffer = fpc2
      f = f2
      feq = feq2
      tau0 = tau
      solid_values = '0 1'
    []
    [speed2]
      type = LBMComputeVelocityMagnitude
      buffer = speed2
      velocity = u2
      solid_values = '0 1'
    []
  []

  [Boundary]
    [wall1]
      type = LBMBounceBack
      buffer = f1
      f_old = fpc1
      boundary = wall
      solid_values = '0 2'
    []
    [wall2]
      type = LBMBounceBack
      buffer = f2
      f_old = fpc2
      boundary = wall
      solid_values = '0 1'
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = 'f1 f2'
  f_old = 'fpc1 fpc2'
[]

[Problem]
  type = LatticeBoltzmannProblem
  scalar_constant_names = 'tau g1 g2'
  scalar_constant_values = '0.8 1e-4 -1e-4'
  substeps = 5
  print_debug_output = true
  binary_media = binary_media
  residual_tensor = speed1
[]

[Postprocessors]
  # mass of each fluid; must be conserved
  [rho1_avg]
    type = TensorAveragePostprocessor
    buffer = rho1
  []
  [rho2_avg]
    type = TensorAveragePostprocessor
    buffer = rho2
  []
  [rho1_max]
    type = TensorExtremeValuePostprocessor
    buffer = rho1
    value_type = MAX
  []
  [rho2_max]
    type = TensorExtremeValuePostprocessor
    buffer = rho2
    value_type = MAX
  []
  # fluids move in opposite directions
  [u1_min]
    type = TensorExtremeValuePostprocessor
    buffer = u1
    value_type = MIN
  []
  [u1_max]
    type = TensorExtremeValuePostprocessor
    buffer = u1
    value_type = MAX
  []
  [u2_min]
    type = TensorExtremeValuePostprocessor
    buffer = u2
    value_type = MIN
  []
  [u2_max]
    type = TensorExtremeValuePostprocessor
    buffer = u2
    value_type = MAX
  []
  [speed1_max]
    type = TensorExtremeValuePostprocessor
    buffer = speed1
    value_type = MAX
  []
  [speed2_max]
    type = TensorExtremeValuePostprocessor
    buffer = speed2
    value_type = MAX
  []
[]

[Executioner]
  type = Transient
  num_steps = 5
[]

[Outputs]
  file_base = two_fluid
  csv = true
[]
