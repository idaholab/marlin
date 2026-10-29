# Boundary condition values that vary along the face, from functions and buffers.
#
# u = (x + 1) * y is harmonic and linear in the normal direction of every face, so the ghost
# values are exact, and the discrete Laplacian must be zero everywhere, including next to the
# faces. Each face takes its value from a different source.
[Domain]
  dim = 2
  nx = 20
  ny = 10
  parallel_mode = REAL_SPACE
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
[]

[Functions]
  [u_func]
    type = ParsedFunction
    expression = '(x + 1) * y'
  []
  [right_value]
    # u on x = 1
    type = ParsedFunction
    expression = '2 * y'
  []
  [top_flux]
    # du/dy on y = 1
    type = ParsedFunction
    expression = 'x + 1'
  []
  [bottom_flux]
    # -du/dy on y = 0 (outward normal -y)
    type = ParsedFunction
    expression = '-(x + 1)'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = MooseFunctionTensor
      buffer = u
      function = u_func
    []
    # buffer values are face values: these depend on the tangential coordinate only
    [bottom_flux]
      type = MooseFunctionTensor
      buffer = bottom_flux
      function = bottom_flux
    []
  []
  [BCs]
    [left]
      # evaluated at the face position x = 0, where it is y
      type = TensorDirichletBC
      variable = u
      boundary = left
      value_function = u_func
    []
    [right]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value_buffer = right_value
    []
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux_buffer = bottom_flux
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux_function = top_flux
    []
  []
  [Solve]
    [lap_compute]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = lap
    []
    # listed after the Laplacian, whose ghost fill reads it: the dependency sort runs it first
    [right_value]
      type = MooseFunctionTensor
      buffer = right_value
      function = right_value
    []
  []
[]

[Postprocessors]
  [max_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MAX
  []
  [min_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MIN
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
