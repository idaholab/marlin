[Domain]
  dim = 3
  nx = 20
  ny = 10
  nz = 5
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'sin(x*8*pi)+cos(y*4*pi)+sin(z*2*pi)'
    []
    [grad_c]
      type = NEML2GradientVector
      buffer = grad_c
      input = c
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
