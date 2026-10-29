# A stencil reaching four cells across a face (the 5-point biharmonic) with Neumann faces in x:
# its four ghost layers are the even mirror of the owned cells about the face. The result on
# 0 < x < 16 then equals the same stencil on the periodic domain 0 < x < 32 of the mirrored field
# (deep_ghost_periodic.i), which needs no ghost fill at a face, to round-off.
#
# u = exp(cos(pi x / 16)) sin(2 pi y / 8) is even about x = 0 and x = 16. With Dirichlet faces and
# u_ic = sin(pi x / 16) exp(cos(2 pi y / 8)), odd about both, the ghosts are the odd mirror.

u_ic = 'exp(cos(pi*x/16))*sin(2*pi*y/8)'

[Domain]
  dim = 2
  nx = 16
  ny = 8
  xmax = 16
  ymax = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
[]

[TensorBuffers]
  [u]
  []
  [bih]
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
  [BCs]
    [left]
      type = TensorNeumannBC
      variable = u
      boundary = left
    []
    [right]
      type = TensorNeumannBC
      variable = u
      boundary = right
    []
  []
  [Postprocess]
    [bih]
      type = FiniteDifferenceBiharmonic
      buffer = bih
      input = u
      stencil_width = 5
    []
    [bih_sq]
      type = ParsedCompute
      buffer = bih_sq
      expression = 'bih^2'
      inputs = bih
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
    buffer = bih
    value_type = MAX
  []
  [min_bih]
    type = TensorExtremeValuePostprocessor
    buffer = bih
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
