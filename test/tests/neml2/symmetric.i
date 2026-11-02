[Domain]
  dim = 3
  nx = 2
  ny = 2
  nz = 2
  mesh_mode = DUMMY
[]

[Problem]
  type = TensorProblem
[]

[TensorBuffers]
  [symmetric]
    type = SR2Tensor
  []
[]

[TensorComputes]
  [Initialize]
    [identity]
      type = RankTwoIdentity
      buffer = identity
    []
    [symmetric]
      type = NEML2TensorCompute
      neml2_input_file = neml2_input.i
      neml2_model = symmetric
      marlin_inputs = identity
      neml2_inputs = input
      neml2_outputs = output
      marlin_outputs = symmetric
    []
  []
[]

[Postprocessors]
  [mean]
    type = TensorAveragePostprocessor
    buffer = symmetric
    execute_on = TIMESTEP_END
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
