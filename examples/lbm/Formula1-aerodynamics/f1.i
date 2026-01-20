[Domain]
  dim = 3
  nx = 781
  ny = 274
  nz = 146
  xmax = 781
  ymax = 274
  zmax = 146
  device_names='cuda'
  floating_precision = 'single'
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Stencil]
  [descriptor]
    type = LBMD3Q27
  []
[]

[TensorBuffers]
  [binary_media]
    type = LBMTensorBuffer
    buffer_type = ms
    file = 'binary_media.h5'
    is_integer = true
  []

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
  [u]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [speed]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [rho]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [force]
    type = LBMTensorBuffer
    buffer_type = mv
  []
[]

[TensorComputes]
  [Initialize]
    [density_initial]
      type = LBMConstantTensor
      buffer = rho
      constants = 1.0
    []

    [velocity_initial]
      type = LBMConstantTensor
      buffer = u
      constants = '0 0 0'
    []

    [equilibrium_init]
      type = LBMEquilibrium
      buffer = feq
      bulk = rho
      velocity = u
    []

    [equilibrium_f]
      type = LBMEquilibrium
      buffer = f
      bulk = rho
      velocity = u
    []

    [equilibrium_pc]
      type = LBMEquilibrium
      buffer = fpc
      bulk = rho
      velocity = u
    []
  []

  [Solve]
    [density]
      type = LBMComputeDensity
      buffer = rho
      f = f
    []

    [velocity]
      type = LBMComputeVelocity
      buffer = u
      f = f
      rho = rho
    []

    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = rho
      velocity = u
    []

    [collision]
      type = LBMSmagorinskyCollision
      buffer = fpc
      f = f
      feq = feq
      tau0 = 0.5001
      Cs = 0.15
      projection=true
    []

    [speed]
      type=LBMComputeVelocityMagnitude
      buffer=speed
      velocity=u
    []

    [residual]
      type = LBMComputeResidual
      buffer = speed
      speed = speed
    []
  []

  [Boundary]
    [wall]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = wall
    []

    [left]
      type = LBMFixedFirstOrderBC
      buffer=f
      f=f
      value=0.01
      boundary=left
      # perturb=true
    []

    [right]
      type = LBMMicroscopicZeroGradientBC
      buffer=f
      boundary = right
    []

  []
[]

[TensorSolver]
  type = LBMStream
  root_compute=residual
  buffer = f
  f_old = fpc
[]

[Postprocessors]
  [reynolds]
    type = ComputeReynoldsNumber
    buffer = speed
    tau = 0.5001
    diameter = 781
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 100
  print_debug_output = true
  is_binary_media = true
  binary_media = binary_media
[]

[Executioner]
  type = Transient
  num_steps = 2000
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'rho u'
    output_mode = 'Cell Cell'
    enable_hdf5 = true
  []
[]
