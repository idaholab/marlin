[Domain]
  # Test parallel FFT round-trip with slab decomposition in 3D
  device_names = "cpu cpu cpu"
  device_weights = "1 1 1"

  parallel_mode = FFT_SLAB

  dim = 3
  nx = 64
  ny = 64
  nz = 64
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
[]

[TensorBuffers]
  [eta_gold]
  []
  [eta]
  []
  [eta_bar]
  []
  [eta_roundtrip]
  []
  [diff]
  []
  [zero]
  []
[]

[TensorComputes]
  [Initialize]
    [eta_gold]
      type = ParsedCompute
      buffer = eta_gold
      expression = 'sin(x)+sin(y)+sin(z)+cos(2*x)*sin(3*y)*cos(z)'
      extra_symbols = true
    []
    [eta]
      type = ParsedCompute
      buffer = eta
      expression = eta_gold
      inputs = eta_gold
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
      real = 0
      imaginary = 0
    []
  []

  [Solve]
    # Test: eta -> FFT -> iFFT -> eta_roundtrip
    # eta_roundtrip should equal eta (within numerical precision)
    [eta_bar]
      type = ForwardFFT
      buffer = eta_bar
      input = eta
    []
    [eta_roundtrip]
      type = InverseFFT
      buffer = eta_roundtrip
      input = eta_bar
    []
  []

  [Postprocess]
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(eta - eta_roundtrip) + abs(eta - eta_gold)'
      inputs = 'eta eta_roundtrip eta_gold'
    []
  []
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
  [l2_error]
    type = TensorIntegralPostprocessor
    buffer = diff
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = eta
  reciprocal_buffer = eta_bar
  linear_reciprocal = zero
  nonlinear_reciprocal = zero
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
  execute_on = 'INITIAL TIMESTEP_END'
[]
