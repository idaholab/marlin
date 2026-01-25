[Domain]
  dim = 3
  nx = 270
  ny = 270
  nz = 405
  xmax = 270
  ymax = 270
  zmax = 405
  device_names='cpu'
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y Z'
  floating_precision=SINGLE
[]

[Stencil]
  [d3q19]
    type = LBMD3Q19
  []
[]

[TensorBuffers]
  # Simulation binary media
  [binary_media]
    type = LBMTensorBuffer
    file = binary_media.h5
    buffer_type = ms
    is_integer = true
  []

  # Density distribution functions
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
  # Temperature distribution functions
  [g]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [geq]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [gpc]
    type = LBMTensorBuffer
    buffer_type = df
  []
  # Fluid macroscopic variables: density and velocity
  [density]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [speed]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  # Temperature macroscpic variables: temperature and 'velocity'
  [T]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  # Forces
  [F]
    type = LBMTensorBuffer
    buffer_type = mv
  []
[]

[TensorComputes]
  [Initialize]
    [density]
      type = LBMConstantTensor
      buffer = density
      constants = 'rho0'
    []
    [velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0 0.0 0.0'
    []
    [temperature]
      type = LBMConstantTensor
      buffer = T
      constants = T_C
    []
    [equilibrium_fluid]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [equilibrium_fluid_total]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
    [equilibrium_fluid_pc]
      type = LBMEquilibrium
      buffer = fpc
      bulk = density
      velocity = velocity
    []
    [equilibrium_temperature]
      type = LBMEquilibrium
      buffer = geq
      bulk = T
      velocity = velocity
    []
    [equilibrium_temperature_total]
      type = LBMEquilibrium
      buffer = g
      bulk = T
      velocity = velocity
    []
    [equilibrium_temperature_pc]
      type = LBMEquilibrium
      buffer = gpc
      bulk = T
      velocity = velocity
    []
  []

  #### Compute ####
  [Solve]

    # For temperature
    [Temperature]
      type = LBMComputeDensity
      buffer = T
      f = g
    []

    # For fluid
    [Fluid_density]
      type = LBMComputeDensity
      buffer = density
      f = f
    []

    [Fluid_velocity]
      type = LBMComputeVelocity
      buffer = velocity
      f = f
      rho = density
      forces = F
      enable_forces = true
    []

    # For temperature
    [Equilibrium_temperature]
      type = LBMEquilibrium
      buffer = geq
      bulk = T
      velocity = velocity
    []

    [Collision_temperature]
      type = LBMBGKCollision
      buffer = gpc
      f = g
      feq = geq
      tau0 = tau_T
    []

    # For fluid
    [Compute_forces]
      type = LBMComputeForces
      buffer = F
      rho0 = 'rho0'
      temperature = T
      T0 = 1.00
      enable_buoyancy = true
      gravity = g
      gravity_direction = 2
    []

    [Equilibrium_fluid]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []

    [Collision_fluid]
      type = LBMBGKCollision
      buffer = fpc
      f = f
      feq = feq
      tau0 = tau_f
    []

    [Apply_forces]
      type = LBMApplyForces
      buffer = fpc
      velocity = velocity
      rho = density
      forces = F
      tau0 = tau_f
    []

    [Residual]
      type = LBMComputeResidual
      speed = T
      # TODO this buffer is redundant, but avoids 'missing parameter' error
      buffer = T
    []
  []

  #### Boundary ####
  [Boundary]
    ##### for fluid
    [wall]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = wall
    []
    ##### for temperature
    [heat_source]
      type = LBMNeumannBC
      buffer = g
      f_old = gpc
      feq=geq
      velocity = velocity
      rho = T
      gradient = 0.001
      region_id = 3
      boundary = regional
    []
    [walls]
      type = LBMDirichletBC
      buffer = g
      f_old = gpc
      feq=geq
      velocity = velocity
      rho = T
      value = 1.0
      region_id = 2
      boundary = regional
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = 'f g'
  f_old = 'fpc gpc'
[]

[Problem]
  type = LatticeBoltzmannProblem
  binary_media = binary_media
  scalar_constant_names = 'rho0 T_C T_H tau_f tau_T g'
  scalar_constant_values = '1.0 1.0 1.05 0.55 0.55 0.01'
  substeps = 100
  print_debug_output = true
[]

[Executioner]
  type = Transient
  num_steps = 1000
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'T density velocity'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
  []
[]
