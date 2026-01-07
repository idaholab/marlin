
[Mesh]
  coord_type = RZ

  [slug]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 20
    ny = 150
    # nx = 20
    # ny = 10
    xmin = 0
    xmax = ${units 0.25 in -> m}
    ymin = ${units -3 in -> m}
    ymax = 0
  []
  [slug_block]
    type = SubdomainIDGenerator
    input = slug
    subdomain_id = 1
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
  large_kinematics = true
[]

[Problem]
  # type = AugmentedLagrangianContactFEProblem
  kernel_coverage_check = false
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
[]

[AuxVariables]
  [vel_x]
    block = 1
  []
  [vel_y]
    block = 1
  []
  [accel_x]
    block = 1
  []
  [accel_y]
    block = 1
  []
  [T]
    initial_condition = ${units 25 degC}
  []
  [ep_dot_recovered]
  []
[]

[UserObjects]
  [ep_dot]
    type = NodalPatchRecoveryMaterialProperty
    property = ep_dot
    execute_on = TIMESTEP_END
    patch_polynomial_order = FIRST
  []
  [terminator]
    type = Terminator
    expression = vel_y<0
    fail_mode = HARD
    execute_on = TIMESTEP_END
  []
[]

[Kernels]
  [sdx]
    type = TotalLagrangianStressDivergenceAxisymmetricCylindrical
    # type = TotalLagrangianStressDivergence
    variable = disp_x
    component = 0
    block = 1
  []
  [sdy]
    type = TotalLagrangianStressDivergenceAxisymmetricCylindrical
    # type = TotalLagrangianStressDivergence
    variable = disp_y
    component = 1
    block = 1
  []

  [ifx]
    type = InertialForce
    variable = disp_x
    velocity = vel_x
    acceleration = accel_x
    beta = 0.25
    gamma = 0.5
    alpha = 0 # not implemented in TotalLagrangianStressDivergence
    block = 1
  []
  [ify]
    type = InertialForce
    variable = disp_y
    velocity = vel_y
    acceleration = accel_y
    beta = 0.25
    gamma = 0.5
    alpha = 0 # not implemented in TotalLagrangianStressDivergence
    block = 1
  []
[]

[AuxKernels]
  [vel_x]
    type = NewmarkVelAux
    variable = vel_x
    acceleration = accel_x
    gamma = 0.5
    execute_on = 'TIMESTEP_END'
  []
  [vel_y]
    type = NewmarkVelAux
    variable = vel_y
    acceleration = accel_y
    gamma = 0.5
    execute_on = 'TIMESTEP_END'
  []

  [accel_x]
    type = NewmarkAccelAux
    variable = accel_x
    displacement = disp_x
    velocity = vel_x
    beta = 0.25
    execute_on = 'TIMESTEP_END'
  []
  [accel_y]
    type = NewmarkAccelAux
    variable = accel_y
    displacement = disp_y
    velocity = vel_y
    beta = 0.25
    execute_on = 'TIMESTEP_END'
  []
  [ep_dot_recovered]
    type = NodalPatchRecoveryAux
    variable = ep_dot_recovered
    nodal_patch_recovery_uo = ep_dot
    execute_on = TIMESTEP_END
  []
[]

[Materials]
  [slug_strain]
    type = ComputeLagrangianStrainAxisymmetricCylindrical
    # type = ComputeLagrangianStrain
    block = 1
  []
  # lag the effective plastic strain rate
  # we might want to node patch recover this to avoid the checkerboarding artifacts
  [effective_plastic_strain_rate]
    type = ParsedMaterial
    property_name = ep_dot
    expression = 'if(dt=0, 1, (ep-ep_old)/dt)'
    material_property_names = 'ep:=effective_plastic_strain ep_old:=Old[effective_plastic_strain]'
    extra_symbols = dt
    outputs = exodus
  []

  [slug_flow_stress]
    type = DerivativeParsedMaterial
    property_name = flow_stress
    # sy   (MPa)
    # K    (MPa)
    # T... (deg C)
    # Pure Al (https://hal.science/hal-03434373/file/PIMM_EJM_2021_SEDDIK.pdf) 10.1016/j.euromechsol.2021.104432
    # expression = '1e8'
    expression = 'sy:=${units 90 MPa -> Pa};
                  K:=${units 200 MPa -> Pa}; C:=0.035;
                  n:=0.3; m:=1; ep_dot_0:=0.01;
                  T_r:=20; T_m:=660.3; T_star:=(T-T_r)/(T_m-T_r); ep_dot_star:=max(1,ep_dot_recovered/ep_dot_0);
                  (sy+K*if(ep>0, ep^n, 0))*(1+C*log(ep_dot_star))*(1-T_star^m)'
    material_property_names = 'ep:=Old[effective_plastic_strain]'
    coupled_variables = 'T ep_dot_recovered'
    additional_derivative_symbols = 'ep'
    derivative_order = 2
    compute = false
    evalerror_behavior = error
    output_properties = flow_stress
    outputs = exodus
  []
  [slug_stress]
    type = ComputeSimoHughesJ2PlasticityStress
    flow_stress_material = slug_flow_stress
    block = 1
    outputs = exodus
    # relative_tolerance = 1e-06
    output_properties = effective_plastic_strain
  []
  [elasticity_slug]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 70 GPa -> Pa}
    poissons_ratio = 0.28
    block = 1
  []
  [slug_density]
    type = StrainAdjustedDensity
    strain_free_density = ${units 2700 kg/m^3}
  []
[]

[ICs]
  [vel_y]
    type = ConstantIC
    variable = vel_y
    value = ${units 200 m/s}
    block = 1
  []
[]

[Functions]
  [anvil]
    type = ParsedFunction
    expression = 'if(y>0, -y*1e15, 0)'
  []
[]

[BCs]
  [anvil]
    type = FunctionNeumannBC
    function = anvil
    variable = disp_y
    boundary = top
    use_displaced_mesh = true
  []
  [symmetry]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = left
  []
[]

[Postprocessors]
  [vel_y]
    type = ElementAverageValue
    variable = vel_y
    block = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [gap_y]
    type = SideAverageValue
    variable = disp_y
    boundary = top
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [rear_y]
    type = SideAverageValue
    variable = disp_y
    boundary = bottom
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [max_ep]
    type = ElementExtremeMaterialProperty
    mat_prop = effective_plastic_strain
    value_type = max
    block = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  dt = ${units 1e-8 s}
[]

[Outputs]
  exodus = true
  # vtk = true
  csv = true
  print_linear_residuals = false
  perf_graph = true
  time_step_interval = 10
[]
