[Domain]
  dim = 3
  nx = 16
  ny = 16
  nz = 16
  xmax = ${fparse 2*pi}
  ymax = ${fparse 2*pi}
  zmax = ${fparse 2*pi}
  mesh_mode = DUMMY
  parallel_mode = FFT_AUTO
[]

[TensorComputes]
  [Initialize]
    [phase]
      type = ParsedCompute
      expression = '(cos(x)/2+0.5)^1*(cos(y)/2+0.5)^1*(cos(z)/2+0.5)^1'
      extra_symbols = true
      buffer = phase
    []
    [K]
      type = ParsedCompute
      buffer = K
      expression = '(1-phase)*Ka + phase*Kb'
      inputs = phase
      constant_names = 'Ka Kb'
      constant_expressions = '1 10'
    []
    [mu]
      type = ParsedCompute
      buffer = mu
      expression = '(1-phase)*mua + phase*mub'
      inputs = phase
      constant_names = 'mua mub'
      constant_expressions = '0.5 5'
    []
    [Finit]
      type = RankTwoIdentity
      buffer = F
    []
  []

  [Solve]
    [hyper_elasticity]
      type = HyperElasticIsotropic
      buffer = stress
      F = Fnew
      K = K
      mu = mu
    []

    [root]
      [applied_strain]
        type = MacroscopicShearTensor
        buffer = applied_strain
      []
      [mech]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        l_tol = 1e-2
        nl_rel_tol = 2e-2
        nl_abs_tol = 2e-2
        constitutive_model = hyper_elasticity
        stress = stress
        applied_macroscopic_strain = applied_strain
      []
    []
  []

  [Postprocess]
    [displacements]
      type = ComputeDisplacements
      buffer = disp
      F = F
    []
    [vonmises]
      type = ComputeVonMisesStress
      buffer = sV
    []
  []
[]

[TensorSolver]
  # no variables are integrated by this solver (FFTMechanics performs a steady state mechanics solve)
  type = ForwardEulerSolver
  root_compute = root
  # deformation tensor is just forwarded Fnew -> F
  forward_buffer = F
  forward_buffer_new = Fnew
  substeps = 10
[]

[TensorOutputs]
  active = 'parallel' # serial

  [serial]
    type = XDMFTensorOutput
    file_base = mech3d_serial
    buffer = 'disp sV F phase'
    output_mode = 'OVERSIZED_NODAL CELL CELL NODE'
    enable_hdf5 = true
    execute_on = 'TIMESTEP_END'
  []
  [parallel]
    # NODE and OVERSIZED_NODAL are not yet available in parallel output
    type = XDMFTensorOutput
    file_base = mech3d_parallel
    buffer = 'sV F phase'
    output_mode = 'CELL CELL CELL'
    enable_hdf5 = true
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  num_steps = 3
  dt = 0.01
[]
