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
    [dAdt]
      type = NEML2TensorCompute
      neml2_input_file = neml2_input.i
      neml2_model = rate
      swift_inputs = 'A B'
      neml2_inputs = 'forces/A forces/B'
      neml2_outputs = 'state/dAdt'
      swift_outputs = 'dAdt'
    []
  []
  [Solve]

  []
[]

[Postprocessors]
  [dAdt]
    type = TensorAveragePostprocessor
    buffer = dAdt
  []
[]

[Executioner]
  type = Transient
  num_steps = 5
[]

[Outputs]
  csv = true
[]
