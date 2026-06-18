[Domain]
  dim = 2
  nx = 102
  ny = 102
  xmax = 102
  ymax = 102
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
  [domain]
    type=LBMTensorBuffer
    file = binary_media.h5
    buffer_type = ms
    is_integer = true
  []
  [local_pore]
    type=LBMTensorBuffer
    file = local_pore_size.h5
    buffer_type = ms
    is_integer = false
  []
  [Kn]
    type=LBMTensorBuffer
    file = Kn.h5
    buffer_type = ms
    is_integer = false
  []
  [relaxation_matrix]
    type = LBMTensorBuffer
    buffer_type = df
  []
[]

[TensorComputes]
  [Initialize]
    [initial_density]
      type = LBMConstantTensor
      buffer = density
      constants = 0.2355545440759889
    []
    [initial_velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0 0.0'
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
    [relaxation_matrix_init]
      type = LBMComputeEffectiveRelaxation
      buffer = relaxation_matrix
      local_pore_size = local_pore
      local_Knudsen_number = Kn
      mfp = 7.904614716131531e-10
      dx = 0.122e-9
      A2 = 0.82
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
      body_force_x = 1.0e-8
    []
    [equilibrium]
      type=LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [collision]
      type = LBMMRTCollision
      buffer = fpc
      f = f
      feq = feq
      is_dynamic_relaxation = true
      local_relaxation_matrix = relaxation_matrix
      projection = true
    []
    [speed]
      type = LBMComputeVelocityMagnitude
      buffer = speed
      velocity = velocity
    []
  []
  [Boundary]
    [wall]
      type = LBMSpecularReflectionBoundary
      buffer = f
      f_old = fpc
      local_Knudsen_number = Kn
      boundary = wall
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = fpc
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = speed
    value_type = MAX
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 2000
  print_debug_output = true
  binary_media = domain
  residual_tensor = speed
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'density velocity speed'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
    transpose=false
  []
[]
