[Domain]
  dim = 3
  nx = 10
  ny = 10
  nz = 10
  mesh_mode=DUMMY
[]

[Stencil]
  [d2q9]
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
  [T]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [F]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [binary_media]
    type = LBMTensorBuffer
    buffer_type = ms
    file = 'input_media.h5'
    is_integer = true
  []
[]

[TensorComputes]

  #### Initialzie ####
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

    [Temperature]
      type = LBMComputeDensity
      buffer = T
      f = g
    []

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

    [Equilibrium_temperature]
      type = LBMEquilibrium
      buffer = geq
      bulk = T
      velocity = velocity
    []

    [Collision_temperature]
      type = LBMMRTCollision
      buffer = gpc
      f = g
      feq = geq
      tau0 = tau_T
    []

    [Compute_forces]
      type = LBMComputeForces
      buffer = F
      rho0 = 'rho0'
      temperature = T
      T0 = T_H
      enable_buoyancy = true
      gravity = g
      gravity_direction=0
    []

    [Equilibrium_fluid]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []

    [Collision_fluid]
      type = LBMMRTCollision
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

    [speed]
      type=LBMComputeVelocityMagnitude
      buffer=speed
      velocity=velocity
    []

    [residual]
      type = LBMComputeResidual
      buffer = speed
      speed = speed
    []
  []

  #### Boundary ####
  [Boundary]
    ##### for fluid
    [inlet]
      type = LBMFixedFirstOrderBC
      buffer = f
      f = f
      value = u0
      boundary = left
    []
    [outlet]
      type = LBMMicroscopicZeroGradientBC
      buffer = f
      boundary = right
    []
    [top]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = top
      # exclude_corners_x = true
    []
    [bottom]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = bottom
      # exclude_corners_x = true
    []
    [front]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = front
      # exclude_corners_x = true
    []
    [back]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = back
      # exclude_corners_x = true
    []
    [wall]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = wall
    []

    ##### for temperature
    [t_inlet]
      type = LBMFixedZerothOrderBC
      buffer = g
      f = g
      value = T_C
      boundary = left
    []
    [t_outlet]
      type = LBMMicroscopicZeroGradientBC
      buffer = g
      boundary = right
    []
    [t_top]
      type = LBMBounceBack
      buffer = g
      f_old = gpc
      boundary = top
    []
    [t_bottom]
      type = LBMBounceBack
      buffer = g
      f_old = gpc
      boundary = bottom
    []
    [t_front]
      type = LBMBounceBack
      buffer = g
      f_old = gpc
      boundary = front
    []
    [t_back]
      type = LBMBounceBack
      buffer = g
      f_old = gpc
      boundary = back
    []
    [t_wall_hot]
      type = LBMDirichletWallBC
      buffer = g
      f_old = gpc
      value = 1.2
      velocity = velocity
      boundary = wall
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
  substeps = 10
  scalar_constant_names = 'rho0 T_C T_H tau_f tau_T g u0'
  scalar_constant_values = '1.0 1.0 1.2 0.9 0.9 0.001 0.001'
  binary_media = binary_media
[]

[Postprocessors]
  [reynolds]
    type = ComputeReynoldsNumber
    buffer = speed
    tau = tau_f
    diameter = 15
  []
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'T velocity density'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
  []
[]
