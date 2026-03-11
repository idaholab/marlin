[Domain]
  dim = 2
  nx = 10
  ny = 10
  mesh_mode = DUMMY
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
      constants = '0.0 0.0'
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
    [left]
      type = LBMNonEquilibriumExtrapolation
      buffer = f
      prescribe_type = velocity
      ux = 0.005
      uy = 0.0
      order = first
      boundary = left
    []
    [right]
      type = LBMConvectiveOutflow
      buffer = f
      f_old = f
      convection_velocity = auto
      boundary = right
    []
    [top]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = top
    []
    [bottom]
      type = LBMBounceBack
      buffer = f
      f_old = fpc
      boundary = bottom
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
  file_base = convective_outflow_2d_right
  csv = true
[]
