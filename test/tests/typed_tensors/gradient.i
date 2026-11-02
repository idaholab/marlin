[Domain]
  dim = 3
  nx = 20
  ny = 10
  nz = 5
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorBuffers]
  [grad_c]
    type = VectorTensor
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'sin(x*8*pi)+cos(y*4*pi)+sin(z*2*pi)'
    []
    [gradient]
      type = NEML2GradientVector
      buffer = gradient
      input = c
    []
    [grad_c]
      type = NEML2TensorCompute
      neml2_input_file = ../neml2/neml2_input.i
      neml2_model = copy_vector
      marlin_inputs = gradient
      neml2_inputs = input
      neml2_outputs = output
      marlin_outputs = grad_c
    []
  []
[]

[Problem]
  type = TensorProblem
  print_debug_output = true
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c grad_c'
    file_base = gradient
    output_mode = 'NODE NODE'
    enable_hdf5 = true
  []
[]
