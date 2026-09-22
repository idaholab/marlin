# 3D version: two grains share order parameter eta0 within the exclusion
# distance and one of them must be moved to a free order parameter.

[Domain]
  dim = 3
  nx = 32
  ny = 32
  nz = 32
  xmax = 32
  ymax = 32
  zmax = 32
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [eta0]
  []
  [eta1]
  []
  [eta2]
  []
[]

[TensorComputes]
  [Initialize]
    [eta0_ic]
      type = ParsedCompute
      buffer = eta0
      extra_symbols = true
      expression = 'exp(-((x-10)^2+(y-16)^2+(z-16)^2)/4.5) + exp(-((x-18)^2+(y-16)^2+(z-16)^2)/4.5)'
    []
    [eta1_ic]
      type = ParsedCompute
      buffer = eta1
      extra_symbols = true
      expression = 'exp(-((x-26)^2+(y-8)^2+(z-8)^2)/4.5)'
    []
    [eta2_ic]
      type = ParsedCompute
      buffer = eta2
      extra_symbols = true
      expression = 'exp(-((x-6)^2+(y-26)^2+(z-26)^2)/4.5)'
    []
  []

  [Postprocess]
    [tracker]
      type = GrainTracker
      op_buffers = 'eta0 eta1 eta2'
      threshold = 0.1
      halo_width = 3
      connectivity = full
      on_conflict = error
    []
  []
[]

[Problem]
  type = TensorProblem
[]

[Postprocessors]
  [grains]
    type = GrainTrackerPostprocessor
    grain_tracker = tracker
    value_type = count
  []
  [remapped]
    type = GrainTrackerPostprocessor
    grain_tracker = tracker
    value_type = remapped
  []
[]

[Executioner]
  type = Transient
  num_steps = 2
  dt = 1
[]

[Outputs]
  csv = true
[]
