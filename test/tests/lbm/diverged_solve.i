[Domain]
  dim = 2
  nx = 1
  ny = 1
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
  [fake]
    type = LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Solve]
    [residual]
      type = LBMNanResidual
      buffer = fake
      step = 42
    []
  []
[]

[TensorSolver]
  type = ForwardEulerSolver
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 10
[]

[Executioner]
  type = Transient
  num_steps = 11
  dtmin = 1
  error_on_dtmin = false
[]
