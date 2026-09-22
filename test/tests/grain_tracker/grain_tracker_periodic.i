# Two grains on eta0 are adjacent only across the periodic x boundary; the
# tracker must detect the conflict through the wrap and recolor one of them.

[Domain]
  dim = 2
  nx = 64
  ny = 64
  xmax = 64
  ymax = 64
  mesh_mode = DUMMY
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [eta0]
  []
  [eta1]
  []
[]

[TensorComputes]
  [Initialize]
    [eta0_ic]
      type = ParsedCompute
      buffer = eta0
      extra_symbols = true
      expression = 'exp(-((x-4)^2+(y-32)^2)/4.5) + exp(-((x-60)^2+(y-32)^2)/4.5)'
    []
    [eta1_ic]
      type = ParsedCompute
      buffer = eta1
      expression = '0'
      expand = REAL
    []
  []

  [Postprocess]
    [tracker]
      type = GrainTracker
      op_buffers = 'eta0 eta1'
      threshold = 0.1
      halo_width = 3
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
