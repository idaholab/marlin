[Domain]
  dim = 2
  nx = 11
  ny = 11
  mesh_mode=DUMMY
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
  [T]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [binary_media]
    type = LBMTensorBuffer
    file = binary_regional.h5
    is_integer = true
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [init_T]
      type = LBMConstantTensor
      buffer = T
      constants = 1.0
    []
    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = T
      velocity = velocity
    []
    [non_equilibrium]
      type = LBMEquilibrium
      buffer = f
      bulk = T
      velocity = velocity
    []
    [post_collision_equilibrium]
      type = LBMEquilibrium
      buffer = fpc
      bulk = T
      velocity = velocity
    []
  []
  [Solve]
    [T]
      type = LBMComputeDensity
      buffer = T
      f = f
    []
  []
  [Boundary]
    [regional]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = T
      value = 1.1
      region_id = 2
      boundary = regional
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
  binary_media = binary_media
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'T'
    output_mode = 'Cell'
    enable_hdf5 = true
  []
[]
