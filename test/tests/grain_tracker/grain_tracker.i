# Two grains share order parameter eta0 and sit closer than the exclusion
# distance; the tracker must move one of them to eta1. A third grain on eta1 is
# far away and must remain untouched.

[Domain]
  dim = 2
  nx = 64
  ny = 64
  xmax = 64
  ymax = 64
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [eta0]
  []
  [eta1]
  []
  [grain_id]
  []
[]

[TensorComputes]
  [Initialize]
    [eta0_ic]
      type = ParsedCompute
      buffer = eta0
      extra_symbols = true
      expression = 'exp(-((x-16)^2+(y-32)^2)/8) + exp(-((x-28)^2+(y-32)^2)/8)'
    []
    [eta1_ic]
      type = ParsedCompute
      buffer = eta1
      extra_symbols = true
      expression = 'exp(-((x-52)^2+(y-10)^2)/8)'
    []
  []

  [Postprocess]
    [tracker]
      type = GrainTracker
      op_buffers = 'eta0 eta1'
      threshold = 0.1
      halo_width = 4
      grain_id_buffer = grain_id
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
  [conflicts]
    type = GrainTrackerPostprocessor
    grain_tracker = tracker
    value_type = conflicts
  []
  [int_eta0]
    type = TensorIntegralPostprocessor
    buffer = eta0
  []
  [int_eta1]
    type = TensorIntegralPostprocessor
    buffer = eta1
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
