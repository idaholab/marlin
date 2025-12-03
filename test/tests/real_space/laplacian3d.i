[Domain]
  dim = 3
  nx = 30
  ny = 24
  nz = 17
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y Z'
  xmin = -1
  ymin = -1
  zmin = -1
  xmax = 1
  ymax = 1
  zmax = 1
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
      expression = 'sin(pi*(x+0.5))*sin(pi*(y+0.5))*sin(pi*(z+0.5))'
      extra_symbols = true
    []
    [lap_gold_ic]
      type = ParsedCompute
      buffer = lap_gold
      #expression = '0'
      expression = '-3*pi*pi*sin(pi*(x+0.5))*sin(pi*(y+0.5))*sin(pi*(z+0.5))'
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
  # active = ''
  [lap]
    type = XDMFTensorOutput
    buffer = 'lap lap_gold diff'
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
