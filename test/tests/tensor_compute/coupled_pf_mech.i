[Domain]
  dim = 3
  nx = 128
  ny = 128
  nz = 128
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
  mesh_mode = DUMMY
[]

[TensorBuffers]
  # phase field
  [c]
  []
  [cbar]
  []
  [mu]
  []
  [mubar]
  []
  [Mbarmubar]
  []

  # mechanics
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
  [mumechbar]
  []
  [mumech]
  []

  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c disp_x disp_y disp_z mu mumech'
    output_mode = 'Node Node Node Node Cell Cell'
    enable_hdf5 = true
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = RandomTensor
      buffer = c
      min = 0.44
      max = 0.56
    []
    [disp_x]
      type = RandomTensor
      buffer = disp_x
      min = 0
      max = 0
    []
    [disp_y]
      type = RandomTensor
      buffer = disp_y
      min = 0
      max = 0
    []
    [disp_z]
      type = RandomTensor
      buffer = disp_z
      min = 0
      max = 0
    []
    [Mbar]
      type = ReciprocalLaplacianFactor
      factor = 0.2 # Mobility
      buffer = Mbar
    []
    [kappabarbar]
      type = ReciprocalLaplacianSquareFactor
      factor = -0.001 # kappa
      buffer = kappabarbar
    []
  []

  [Solve]
    [mu]
      # chemical potential (real space)
      type = ParsedCompute
      buffer = mu
      expression = '0.1*c^2*(c-1)^2' # + c*sin(x/2)*0.005'
      extra_symbols = true
      derivatives = c
      inputs = c
    []
    [mubar]
      # chemical potential (reciprocal space)
      type = ForwardFFT
      buffer = mubar
      input = mu
    []
    [mumechbar]
      # mechanical chemical potential (reciprocal space)
      type = FFTElasticChemicalPotential
      buffer = mumechbar
      cbar = cbar
      displacements = 'disp_x disp_y disp_z'
      lambda = 100
      mu = 50
      e0 = 0.02
    []
    [mumech]
      # chemical potential (reciprocal space)
      type = InverseFFT
      buffer = mumech
      input = mumechbar
    []

    [Mbarmubar]
      type = ParsedCompute
      buffer = Mbarmubar
      expression = 'Mbar*(mubar+mumechbar)'
      inputs = 'Mbar mubar mumechbar'
    []
    [cbar]
      type = ForwardFFT
      buffer = cbar
      input = c
    []

    [qsmech]
      type = FFTQuasistaticElasticity
      displacements = 'disp_x disp_y disp_z'
      cbar = cbar
      lambda = 100
      mu = 50
      e0 = 0.02
    []
  []
[]

[TensorTimeIntegrators]
  [c]
    type = FFTSemiImplicit
    buffer = c
    reciprocal_buffer = cbar
    linear_reciprocal = kappabarbar
    nonlinear_reciprocal = Mbarmubar
  []
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [min_disp_x]
    type = TensorExtremeValuePostprocessor
    buffer = disp_x
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_x]
    type = TensorExtremeValuePostprocessor
    buffer = disp_x
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_y]
    type = TensorExtremeValuePostprocessor
    buffer = disp_y
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_y]
    type = TensorExtremeValuePostprocessor
    buffer = disp_y
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_z]
    type = TensorExtremeValuePostprocessor
    buffer = disp_z
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_z]
    type = TensorExtremeValuePostprocessor
    buffer = disp_z
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [C]
    type = TensorIntegralPostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
  [cavg]
    type = TensorAveragePostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 1000
  print_debug_output = true
[]

[Executioner]
  type = Transient
  num_steps = 100
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.8
    dt = 0.1
  []
  dtmax = 1000
[]

[Outputs]
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
