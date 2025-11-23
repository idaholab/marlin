[Domain]
  dim = 2
  nx = 16
  ny = 16
  parallel_mode = REAL_SPACE
  ghost_layers = 1
  xmin = -1
  ymin = -1
  xmax = 1
  ymax = 1
[]

[TensorBuffers]
  [u]
    type = PlainTensorBuffer
  []
  [u_gold]
    type = PlainTensorBuffer
  []
  [lap]
    type = PlainTensorBuffer
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      expression = 'sin(pi*x)*sin(pi*y)'
      extra_symbols = true
    []
    [lap_gold_ic]
      type = ParsedCompute
      buffer = lap_gold
      #expression = '0'
      expression = '-pi*pi*sin(pi*x)*sin(pi*y)'
      extra_symbols = true
      expand = REAL
    []
  []
  [Solve]
    [lap_compute]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = lap
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(lap - lap_gold)'
      inputs = 'lap lap_gold'
    []
  []
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
[]

[TensorOutputs]
  [lap]
    type = XDMFTensorOutput
    buffer = 'lap lap_gold'
    enable_hdf5 = true
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
