# The reference of deep_ghost_faces.i: the field mirrored onto the periodic domain 0 < x < 32,
# where the 5-point biharmonic reads only periodic ghosts. The postprocessors take 0 < x < 16.

u_ic = 'exp(cos(pi*x/16))*sin(2*pi*y/8)'

[Domain]
  dim = 2
  nx = 32
  ny = 8
  xmax = 32
  ymax = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [u]
  []
  [bih]
  []
  [bih_strip]
  []
  [bih_sq]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      extra_symbols = true
      expression = ${u_ic}
    []
  []
  [Postprocess]
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih
      input = u
      stencil_width = 5
    []
    # 0 outside the strip, which is between the extremes of both test fields there
    [bih_strip]
      type = ParsedCompute
      buffer = bih_strip
      extra_symbols = true
      expression = 'if(x < 16, bih, 0)'
      inputs = bih
    []
    [bih_sq]
      type = ParsedCompute
      buffer = bih_sq
      expression = 'bih_strip^2'
      inputs = bih_strip
    []
  []
[]

[Postprocessors]
  [integral_bih_sq]
    type = TensorIntegralPostprocessor
    buffer = bih_sq
  []
  [max_bih]
    type = TensorExtremeValuePostprocessor
    buffer = bih_strip
    value_type = MAX
  []
  [min_bih]
    type = TensorExtremeValuePostprocessor
    buffer = bih_strip
    value_type = MIN
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
