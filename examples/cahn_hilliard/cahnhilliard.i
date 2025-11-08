#
# Simple Cahn-Hilliard solve on a 2D grid. We create a matching (conforming)
# MOOSE mesh (with one element per FFT grid cell) and project the solution onto
# the MOOSE mesh to utilize the exodus output object.
#

[Domain]
  dim = 2
  nx = 200
  ny = 200

  xmax = ${fparse pi*8}
  ymax = ${fparse pi*8}

  # automatically create a matching mesh
  mesh_mode = DOMAIN
[]


[TensorBuffers]
  [c]
    # perform fast mapping to the matching mesh by directly writing to
    # the solution vector of the specified Auxvariable
    map_to_aux_variable = c
  []
  [cbar]
  []
  [mu]
    map_to_aux_variable = mu
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

[TensorComputes]
  [Initialize]
    [c]
      # Random initial condition around a concentration of 1/2
      type = RandomTensor
      buffer = c
      min = 0.44
      max = 0.56
    []

    # precompute fixed factors for the solve
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
    [mu_init]
      type = ConstantTensor
      buffer = mu
      real = 0
    []
  []

  [Solve]
    [cahn_hilliard]
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
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  root_compute = cahn_hilliard
  buffer = c
  reciprocal_buffer = cbar
  linear_reciprocal = kappabarbar
  nonlinear_reciprocal = Mbarmubar
  substeps = 1000
[]

[AuxVariables]
  [mu]
    # the mu tensor  is projected onto this elemental variable
    family = MONOMIAL
    order = CONSTANT
  []
  [c]
    # the c tensor is projected onto this nodal variable
  []
[]

# a slower but more flexible alternative to `map_to_aux_variable` is running
# these `ProjectTensorAux` AuxKernels to perform the projection. This aprpoach
# also supports non-conforming meshes.
[AuxKernels]
  # [c]
  #   type = ProjectTensorAux
  #   buffer = c
  #   variable = c
  #   execute_on = final
  # []
  # [f]
  #   type = ProjectTensorAux
  #   buffer = f
  #   variable = f
  #   execute_on = TIMESTEP_END
  # []
[]

[Postprocessors]
  [min_c]
    type = ElementExtremeValue
    variable = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = ElementExtremeValue
    variable = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  # [F]
  #   type = ElementIntegralVariablePostprocessor
  #   variable = f
  #   execute_on = 'TIMESTEP_END'
  # []
  [C]
    type = ElementIntegralVariablePostprocessor
    variable = c
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
  exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
