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
    file = binary_media.h5
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
    [wall]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = T
      value = 1.1
      boundary = wall
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
  binary_media = binary_media
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
    buffer = T
    value_type = MIN
  []
  [densty_max]
    type = TensorExtremeValuePostprocessor
    buffer = T
    value_type = MAX
  []
[]

[Executioner]
  type = Transient
  num_steps = 5
[]

[Outputs]
  file_base = dirichlet_wall
  csv = true
[]
