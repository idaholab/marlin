# timestep in seconds
dt=1e-8

# velocity in m/s
v=200

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Problem]
  extra_tag_matrices = 'mass'
[]

# variables
[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]

[AuxVariables]
  [force_x]
  []
  [force_y]
  []
  [force_z]
  []
[]

[AuxKernels]
  [force_x]
    type = TagVectorAux
    v = disp_x
    variable = force_x
    vector_tag = NONTIME
  []
  [force_y]
    type = TagVectorAux
    v = disp_y
    variable = force_y
    vector_tag = NONTIME
  []
  [force_z]
    type = TagVectorAux
    v = disp_z
    variable = force_z
    vector_tag = NONTIME
  []
[]

# cylindrical slug mesh
[Mesh]
  [disc]
    type = ConcentricCircleMeshGenerator
    num_sectors = 8 # adjust for mesh resolution
    radii = ${units 0.25 in -> m}
    rings = 10 # adjust for mesh resolution
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
    heights = '${units 3 in -> m}'
    num_layers = 50 # adjust for mesh resolution
  []
  [face]
    type = SideSetsAroundSubdomainGenerator
    block = 1
    input = extrude
    new_boundary = face
    normal = '0 1 0'
  []
[]

# set velocity in y direction
[ICs]
  [current]
    type = ConstantIC
    variable = disp_y
    value = 0
    state = CURRENT
  []
  [old]
    type = ConstantIC
    variable = disp_y
    value = ${fparse -v*dt}
    state = OLD
  []
[]

# simple penalty anvil BC
[Functions]
  [anvil]
    type = ParsedFunction
    expression = 'if(y>0, -y*${units 20000 GPa -> Pa}, 0)'
  []
[]
[BCs]
  [anvil]
    type = FunctionNeumannBC
    function = anvil
    variable = disp_y
    boundary = face
    use_displaced_mesh = true
  []
[]

# material properties
[Materials]
  # [elasticity_slug]
  #   type = ComputeIsotropicElasticityTensor
  #   youngs_modulus = ${units 70 GPa -> Pa}
  #   poissons_ratio = 0.28
  # []
  [slug_density]
    type = StrainAdjustedDensity
    strain_free_density = ${units 2700 kg/m^3}
  []
[]

# mechanics and mass matrix kernels
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
  input = 'johnson_cook_neml2.i'
  [all]
    executor_name = 'neml2'
    model = 'model'
    verbose = true
    moose_input_kernels = 'strain'

    moose_input_types = 'POSTPROCESSOR POSTPROCESSOR POSTPROCESSOR'
    moose_inputs = '     time          time          temperature'
    neml2_inputs = '     forces/t      old_forces/t  forces/T'
  []
[]

[Postprocessors]
  [time]
    type = TimePostprocessor
    execute_on = 'INITIAL TIMESTEP_BEGIN'
    outputs = 'none'
  []
  [temperature]
    type = ConstantPostprocessor
    value = 300
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
  [strain]
    type = NEML2SmallStrain
    assembly = 'assembly'
    fe = 'fe'
    to_neml2 = 'forces/E'
  []
  [residual]
    type = NEML2StressDivergence
    assembly = 'assembly'
    fe = 'fe'
    executor = 'neml2'
    stress = 'state/S'
    residual = 'NONTIME'
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

  start_time = 0.0
  num_steps = 10000
  dt = ${units ${dt} s}
[]

[Outputs]
  time_step_interval = 50
  exodus = true
[]
