# 3-D Taylor anvil impact on the current production stack: multiplicative
# thermal Johnson-Cook using the calibrated material parameter set,
# reduced integration + batched HEX8 hourglass control, boundary-restricted
# nodal penalty contact (frictionless -- the calibration rejects friction).
# 3-D twin of rz_slug_thermal_mult_ri.i.

# timestep in seconds
dt = 5e-9

# impact velocity in m/s -- CuH04_235.9 shot
v = 235.9

# TRUE specimen geometry: 0.3 in diameter x 1.5 in, full-hard OFHC copper
r = '${units 0.15 in -> m}'
slug_length = '${units 1.5 in -> m}'

# mesh resolution
n_sectors = 4
n_rings = 2
n_layers = 24

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Problem]
  extra_tag_matrices = 'mass'
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]

# elemental state fields for visualization (the NEML2 states are material
# properties; surface them as constant-monomial aux for the Exodus output)
[AuxVariables]
  [ep]
    family = MONOMIAL
    order = CONSTANT
  []
  [dT]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [ep]
    type = MaterialRealAux
    variable = ep
    property = 'state/ep'
    execute_on = TIMESTEP_END
  []
  [dT]
    type = MaterialRealAux
    variable = dT
    property = 'state/dT'
    execute_on = TIMESTEP_END
  []
[]

[Mesh]
  [disc]
    type = ConcentricCircleMeshGenerator
    num_sectors = '${n_sectors}'
    radii = '${r}'
    rings = '${n_rings}'
    has_outer_square = false
    preserve_volumes = true
    smoothing_max_it = 3
  []
  [rotate_x_90]
    type = TransformGenerator
    input = disc
    transform = ROTATE
    vector_value = '0 90 0'
  []
  [extrude]
    type = AdvancedExtruderGenerator
    input = rotate_x_90
    direction = '0 1 0'
    heights = '${slug_length}'
    num_layers = '${n_layers}'
  []
  [impact_face]
    type = SideSetsAroundSubdomainGenerator
    block = 1
    input = extrude
    new_boundary = impact_face
    normal = '0 -1 0'
  []
[]

# initial -y velocity via nonzero OLD state: v_y = (current - old)/dt = -v
[ICs]
  [old]
    type = ConstantIC
    variable = disp_y
    value = '${fparse v*dt}'
    state = OLD
  []
[]

# rigid anvil: node-wise penalty wall on the impact face (frictionless)
[NodalKernels]
  [anvil]
    type = PenaltyRigidWallNodalKernel
    variable = disp_y
    component = 1
    penalty = 1e9
    wall_position = 0
    boundary = impact_face
  []
[]

[Materials]
  [slug_density]
    type = StrainAdjustedDensity
    strain_free_density = '${units 8960 kg/m^3}'
  []
[]

[Kernels]
  [mass_x]
    type = MassMatrix
    density = density
    matrix_tags = 'mass'
    variable = disp_x
  []
  [mass_y]
    type = MassMatrix
    density = density
    matrix_tags = 'mass'
    variable = disp_y
  []
  [mass_z]
    type = MassMatrix
    density = density
    matrix_tags = 'mass'
    variable = disp_z
  []
[]

[NEML2]
  input = 'johnson_cook_neml2_mult_thermal_calibrated.i'
  [all]
    executor_name = 'neml2'
    model = 'model'
    verbose = false
    manage_state_advance = true
    identity_seeded_state = 'state/Fp'
    input_kernels = 'deformation_gradient'
    auto_output = true
  []
[]

[UserObjects]
  [assembly]
    type = NEML2Assembly
  []
  [fe]
    type = NEML2FEInterpolation
    assembly = 'assembly'
  []
  [deformation_gradient]
    type = NEML2DeformationGradient
    assembly = 'assembly'
    fe = 'fe'
    to_neml2 = 'deformation_gradient'
  []
  [residual]
    type = NEML2StressDivergence
    assembly = 'assembly'
    fe = 'fe'
    executor = 'neml2'
    stress = 'pk1'
    residual = 'NONTIME'
  []
  [hourglass]
    type = NEML2HourglassCorrection
    assembly = 'assembly'
    fe = 'fe'
    executor = 'neml2'
    displacements = 'disp_x disp_y disp_z'
    # classic FB coefficient with the physical shear modulus mu = E/(2(1+nu))
    penalty = 0.05
    shear_modulus = 43.7e9
    residual = 'NONTIME'
  []
  # stop as soon as the slug's average axial velocity reverses sign (rebound)
  [rebound]
    type = Terminator
    expression = 'vel_y_avg > 0'
    fail_mode = HARD
    execute_on = TIMESTEP_END
    message = 'Slug COM axial velocity reversed -- rebound detected, stopping.'
  []
[]

[Postprocessors]
  [disp_y_avg]
    type = ElementAverageValue
    variable = disp_y
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [d_disp_y]
    type = ChangeOverTimePostprocessor
    postprocessor = disp_y_avg
    change_with_respect_to_initial = false
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [vel_y_avg]
    type = ParsedPostprocessor
    pp_names = 'd_disp_y'
    pp_symbols = 'dd'
    expression = 'dd / ${dt}'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [max_dT]
    type = ElementExtremeMaterialProperty
    mat_prop = 'state/dT'
    value_type = max
    execute_on = 'TIMESTEP_END'
  []
  [max_ep]
    type = ElementExtremeMaterialProperty
    mat_prop = 'state/ep'
    value_type = max
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  [TimeIntegrator]
    type = NEML2CentralDifference
    mass_matrix_tag = 'mass'
    use_constant_mass = true
    second_order_vars = 'disp_x disp_y disp_z'
    assembly = 'assembly'
    fe = 'fe'
  []
  [Quadrature]
    type = GAUSS
    order = CONSTANT
  []

  start_time = 0.0
  # Keep the checked-in example short; raise this for a full arrest calculation.
  num_steps = 100
  dt = '${units ${dt} s}'
  dtmin = '${units ${dt} s}'
[]

[Outputs]
  [exodus]
    type = Exodus
    time_step_interval = 100
  []
  [csv]
    type = CSV
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [console]
    type = Console
    time_step_interval = 100
  []
[]
