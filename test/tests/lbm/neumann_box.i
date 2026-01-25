[Domain]
  dim = 3
  nx = 5
  ny = 5
  nz = 5
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
  [density]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
[]

[TensorComputes]
  [Initialize]
    [init_density]
      type = LBMConstantTensor
      buffer = density
      constants = 1.0
    []
    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [non_equilibrium]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
    [post_collision_equilibrium]
      type = LBMEquilibrium
      buffer = fpc
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
  []
  [Boundary]
    [left]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.001
      boundary = left
    []
    [right]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.1
      boundary = right
    []
    [top]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.1
      boundary = top
    []
    [bottom]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.1
      boundary = bottom
    []
    [front]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.1
      boundary = front
    []
    [back]
      type = LBMNeumannBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      gradient = 0.1
      boundary = back
    []
  []
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
  file_base = neumann_box
  csv = true
[]
