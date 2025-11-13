[Domain]
  dim = 2
  nx = 2
  ny = 2
  xmax = 1
  ymax = 1
  mesh_mode = DUMMY
[]

[Problem]
  type = TensorProblem
[]

[TensorComputes]
  [Initialize]
    [A]
      type = ConstantTensor
      buffer = A
      real = 2

    []
    [B]
      type = ConstantTensor
      buffer = B
      real = 3
    []
    [C]
      type = NEML2TensorCompute
      neml2_input_file = neml2_input.i
      neml2_model = multiply
      marlin_inputs = 'A B'
      neml2_inputs = 'forces/A forces/B'
      neml2_outputs = 'state/C'
      marlin_outputs = 'C'
    []
  []
[]

[Postprocessors]
  [C]
    type = TensorAveragePostprocessor
    buffer = C
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
