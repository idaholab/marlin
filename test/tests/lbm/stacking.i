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
  [ux]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [uy]
    type=LBMTensorBuffer
    buffer_type = ms
  []
  [u]
    type=LBMTensorBuffer
    buffer_type = mv
  []
[]

[TensorComputes/Initialize]
  [velocity_x]
    type = ParsedCompute
    buffer = ux
    expression = '0.1*sin(x/(2*pi*4))*cos(y/(2*pi*4))'
    extra_symbols=true
  []
  [velocity_y]
    type = ParsedCompute
    buffer = uy
    expression = '-0.1*cos(x/(2*pi*4))*sin(y/(2*pi*4))'
    extra_symbols=true
  []
  [u_stack]
    type=LBMStackTensors
    buffer=u
    inputs='ux uy'
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
[]

[Postprocessors]
  [velocity_x_min]
    type = TensorExtremeValuePostprocessor
    buffer = ux
    value_type = MIN
  []
  [velocity_x_max]
    type = TensorExtremeValuePostprocessor
    buffer = ux
    value_type = MAX
  []
  [velocity_y_min]
    type = TensorExtremeValuePostprocessor
    buffer = uy
    value_type = MIN
  []
  [velocity_y_max]
    type = TensorExtremeValuePostprocessor
    buffer = uy
    value_type = MAX
  []
  [u_stack_min]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
  []
  [u_stack_max]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[Outputs]
  file_base = stacking
  csv = true
[]
