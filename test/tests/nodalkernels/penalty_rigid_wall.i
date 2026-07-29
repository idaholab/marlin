# Elastic block dropped onto a rigid wall enforced by PenaltyRigidWallNodalKernel.
# The block starts 0.005 m above the wall moving down at 5 m/s, bounces, and by
# the end of the run is moving up with (nearly) the incoming speed. Verifies the
# node-wise one-sided penalty force and its interaction with ExplicitMixedOrder.

[Mesh]
  [block]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 2
    ny = 2
    xmin = 0
    xmax = 0.1
    ymin = 0.005
    ymax = 0.105
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Problem]
  extra_tag_matrices = 'mass'
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
[]

# downward velocity via OLD state: v_y = (current - old)/dt = -5 m/s
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
    value = '${fparse 5 * 1e-5}'
    state = OLD
  []
[]

[NodalKernels]
  [wall]
    type = PenaltyRigidWallNodalKernel
    variable = disp_y
    component = 1
    penalty = 1e7
    wall_position = 0
  []
[]

[Kernels]
  [sdx]
    type = StressDivergenceTensors
    variable = disp_x
    component = 0
    use_displaced_mesh = false
  []
  [sdy]
    type = StressDivergenceTensors
    variable = disp_y
    component = 1
    use_displaced_mesh = false
  []
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
[]

[Materials]
  [C]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1e9
    poissons_ratio = 0.3
  []
  [strain]
    type = ComputeSmallStrain
    implicit = false
  []
  [stress]
    type = ComputeLinearElasticStress
    implicit = false
  []
  [density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = 1000
  []
[]

[Postprocessors]
  [disp_y_avg]
    type = ElementAverageValue
    variable = disp_y
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [disp_y_min]
    type = NodalExtremeValue
    variable = disp_y
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  [TimeIntegrator]
    type = ExplicitMixedOrder
    mass_matrix_tag = 'mass'
    use_constant_mass = true
    second_order_vars = 'disp_x disp_y'
  []
  start_time = 0.0
  num_steps = 300
  dt = 1e-5
[]

[Outputs]
  csv = true
[]
