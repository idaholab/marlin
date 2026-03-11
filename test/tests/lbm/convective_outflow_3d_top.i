[Domain]
  dim = 3
  nx = 8
  ny = 8
  nz = 8
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
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [density]
    type = LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [initial_density]
      type = LBMConstantTensor
      buffer = density
      constants = 1.0
    []
    [initial_velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0 0.0 0.0'
    []
    [initial_feq]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [initial_f]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
    [initial_fpc]
      type = LBMEquilibrium
      buffer = fpc
      bulk = density
      velocity = velocity
    []
  []
  [Solve]
    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []
    [collision]
      type = LBMBGKCollision
      buffer = fpc
      f = f
      feq = feq
      tau0 = 1.0
    []
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
    []
  []
  [Boundary]
    [bottom]
      type = LBMNonEquilibriumExtrapolation
      buffer = f
      prescribe_type = velocity
      ux = 0.0
      uy = 0.005
      uz = 0.0
      order = first
      boundary = bottom
    []
    [top]
      type = LBMConvectiveOutflow
      buffer = f
      f_old = f
      convection_velocity = Uc
      boundary = top
    []
    [left]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = left
    []
    [right]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = right
    []
    [front]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = front
    []
    [back]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = back
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = fpc
[]

[Problem]
  type = LatticeBoltzmannProblem
  scalar_constant_names = 'Uc'
  scalar_constant_values = '0.005'
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
  [density_max]
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
  file_base = convective_outflow_3d_top
  csv = true
[]
