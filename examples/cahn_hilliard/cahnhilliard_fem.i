[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 100
  ny = 100
  nz = 100
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
[]

[Modules]
  [./PhaseField]
    [./Conserved]
      [./c]
        free_energy = fbulk
        mobility = M
        kappa = kappa_c
        solve_type = REVERSE_SPLIT
      [../]
    [../]
  [../]
[]

[ICs]
  [./cIC]
    type = RandomIC
    variable = c
    min = 0.39
    max = 0.41
  [../]
[]

[BCs]
  [./Periodic]
    [./all]
      auto_direction = 'x y'
    [../]
  [../]
[]

[Materials]
  [./mat]
    type = GenericConstantMaterial
    prop_names  = 'M kappa_c'
    prop_values = '0.2 0.001'
  [../]
  [./free_energy]
    type = DerivativeParsedMaterial
    property_name = fbulk
    coupled_variables = c
    expression = 0.1*c^2*(c-1)^2
  [../]
[]

[Preconditioning]
  [./cw_coupling]
    type = SMP
    full = true
  [../]
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  scheme = bdf2

  petsc_options_iname = '-pc_type -sub_pc_type'
  petsc_options_value = 'asm      lu          '

  l_max_its = 30
  l_tol = 1e-4
  nl_max_its = 20
  nl_rel_tol = 1e-9

  dt = 1
  num_steps = 100
[]

[Outputs]
  exodus = true
  perf_graph = true
[]
