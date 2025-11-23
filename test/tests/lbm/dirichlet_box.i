[Domain]
  dim = 3
  nx = 5
  ny = 5
  nz = 5
  mesh_mode = DUMMY
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
    [density]
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
  [Boundary]
    [left]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = left
    []
    [right]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = right
    []
    [top]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = top
    []
    [bottom]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq=geq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = bottom
    []
    [front]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = front
    []
    [back]
      type = LBMDirichletBC
      buffer = f
      f_old = fpc
      feq = feq
      velocity = velocity
      rho = density
      value = 1.1
      boundary = back
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'density'
    output_mode = 'Cell'
    enable_hdf5 = true
  []
[]
