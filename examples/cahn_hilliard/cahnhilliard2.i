#
# The same simple Cahn-Hilliard solve as cahnhilliard.i, but on a 3D grid
# and using the faster TensorOutputs system.
#

[Domain]
  dim = 3
  nx = 200
  ny = 200
  nz = 200
  xmax = ${fparse pi*8}
  ymax = ${fparse pi*8}
  zmax = ${fparse pi*8}

  # run on a CUDA device (adjust this to `cpu` if not available)
  device_names = 'cuda'

  # create a single element dummy mesh. Output will use the custom XDMF output
  # in the `TensorOutputs` system.
  mesh_mode = DUMMY
[]

[TensorBuffers]
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
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[TensorOutputs]
  # the TensorOutouts system supports asynchronous threaded output.
  # for GOU calculations a copy of the solution fields is moved to the CPU,
  # and while the output files are written the next time step is already
  # starting to compute.
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c mu'
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
      type = ParsedCompute
      buffer = mu
      expression = '0.1*c^2*(c-1)^2'
      derivatives = c
      inputs = c
    []
    [mubar]
      type = ForwardFFT
      buffer = mubar
      input = mu
    []
    [Mbarmubar]
      type = ParsedCompute
      buffer = Mbarmubar
      expression = 'Mbar*mubar'
      inputs = 'Mbar mubar'
    []
    [cbar]
      type = ForwardFFT
      buffer = cbar
      input = c
    []
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = c
  reciprocal_buffer = cbar
  linear_reciprocal = kappabarbar
  nonlinear_reciprocal = Mbarmubar
  substeps = 1000
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
