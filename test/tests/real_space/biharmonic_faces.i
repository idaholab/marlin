# The biharmonic applied explicitly to u = sin(pi x) sin(pi y) + x^2 - y^2 with Dirichlet faces
# (u = x^2 - y^2 there, curved along the faces), once in [Solve] and once in [Postprocess]: both
# impose u = g and Laplacian(u) = 0 on the faces, so they agree. max b1 matches a NumPy model of
# the two-stage operator (523.01524485377); the 13-point stencil on u's ghosts gives 677.76.
[Domain]
  dim = 2
  nx = 16
  ny = 16
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
[]
[Functions]
  [uf]
    type = ParsedFunction
    expression = 'sin(pi*x)*sin(pi*y) + x^2 - y^2'
  []
  [hf]
    type = ParsedFunction
    expression = 'x^2 - y^2'
  []
[]
[TensorBuffers]
  [u]
  []
  [b1]
  []
  [b2]
  []
  [d]
  []
[]
[TensorComputes]
  [Initialize]
    [u]
      type = MooseFunctionTensor
      buffer = u
      function = uf
    []
  []
  [BCs]
    [l]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value_function = hf
    []
    [r]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value_function = hf
    []
    [b]
      type = TensorDirichletBC
      variable = u
      boundary = bottom
      value_function = hf
    []
    [t]
      type = TensorDirichletBC
      variable = u
      boundary = top
      value_function = hf
    []
  []
  [Solve]
    [bs]
      type = FiniteDifferenceBiharmonic
      buffer = b1
      input = u
    []
  []
  [Postprocess]
    [bp]
      type = FiniteDifferenceBiharmonic
      buffer = b2
      input = u
    []
    [diff]
      type = ParsedCompute
      buffer = d
      expression = 'abs(b1 - b2)'
      inputs = 'b1 b2'
    []
  []
[]
[Postprocessors]
  [maxdiff]
    type = TensorExtremeValuePostprocessor
    buffer = d
    value_type = MAX
  []
  [maxb1]
    type = TensorExtremeValuePostprocessor
    buffer = b1
    value_type = MAX
  []
  [maxb2]
    type = TensorExtremeValuePostprocessor
    buffer = b2
    value_type = MAX
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
