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
  []
  [Solve]
    [time]
      type = TimeTensorCompute
      buffer = time
    []
    [dAdt]
      type = NEML2TensorCompute
      neml2_input_file = neml2_input.i
      neml2_model = rate
      swift_inputs = 'A time'
      neml2_inputs = 'forces/A forces/t'
      neml2_outputs = 'state/dAdt'
      swift_outputs = 'dAdt'
    []
  []
[]

[Postprocessors]
  [dAdt]
    type = TensorAveragePostprocessor
    buffer = dAdt
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  num_steps = 5
[]

[Outputs]
  csv = true
[]
