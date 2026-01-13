[Domain]
  dim = 2
  nx = 11
  ny = 11
  mesh_mode=DUMMY
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [rho]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [u]
    type=LBMTensorBuffer
    buffer_type = mv
  []
  [speed]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [f]
    type=LBMTensorBuffer
    buffer_type = df
  []
  [feq]
    type=LBMTensorBuffer
    buffer_type = df
  []
  [f_post_collision]
    type=LBMTensorBuffer
    buffer_type = df
  []
  [binary_media]
    type = LBMTensorBuffer
    file = binary_media.h5
    is_integer = true
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [rho]
      type=LBMConstantTensor
      buffer=rho
      constants = rho0
    []
    [u]
      type=LBMConstantTensor
      buffer=u
      constants = 'Ux Uy'
    []
    [speed]
      type=LBMComputeVelocityMagnitude
      buffer=speed
      velocity=u
    []
    [feq]
      type=LBMEquilibrium
      buffer=feq
      bulk=rho
      velocity=u
    []
    [f]
      type=LBMEquilibrium
      buffer=f
      bulk=rho
      velocity=u
    []
    [f_post_coll]
      type=LBMEquilibrium
      buffer=f_post_collision
      bulk=rho
      velocity=u
    []
  []
  [Solve]
    [Density]
      type = LBMComputeDensity
      buffer=rho
      f = f
    []
    [Velocity]
      type = LBMComputeVelocity
      buffer=u
      f = f
      rho = rho
    []
    [Equilibrium]
      type = LBMEquilibrium
      buffer=feq
      bulk=rho
      velocity=u
    []
    [Collision]
      type = LBMSmagorinskyCollision
      buffer = f_post_collision
      f = f
      feq = feq
      tau0 = tau
      projection=true
    []
    [Speed]
      type = LBMComputeVelocityMagnitude
      buffer = speed
      velocity = u
    []
    [Residual]
      type = LBMComputeResidual
      buffer = speed
      speed = speed
    []
  []
  [Boundary]
    [wall]
      type = LBMBounceBack
      buffer = f
      f_old = f_post_collision
      boundary = wall
    []
    [left]
      type = LBMFixedFirstOrderBC
      buffer=f
      f=f
      value='Ux'
      perturb=true
      boundary=left
    []
    [right]
      type = LBMMicroscopicZeroGradientBC
      buffer=f
      boundary=right
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = f_post_collision
[]

[Postprocessors]
  [rho_avg]
    type = TensorAveragePostprocessor
    buffer = rho
    execute_on = 'TIMESTEP_END'
  []
  [speed_avg]
    type = TensorAveragePostprocessor
    buffer = speed
    execute_on = 'TIMESTEP_END'
  []
  [reynolds]
    type = ComputeReynoldsNumber
    buffer = speed
    tau = tau
    diameter = D
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  scalar_constant_names = 'rho0 Ux    Uy tau   dx    D    Cs'
  scalar_constant_values = '1.0 0.01 0.0 0.501 0.001 50 0.15'
  substeps = 5
  print_debug_output=true
  binary_media = binary_media
[]

[Executioner]
  type = Transient
  num_steps = 3
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'rho u speed binary_media'
    output_mode = 'Cell Cell Cell Cell'
    enable_hdf5 = true
  []
[]
