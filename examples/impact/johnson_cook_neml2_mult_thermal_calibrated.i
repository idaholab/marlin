# CALIBRATED variant (shot CuH04_235.9 profile fit: strength x0.90, n=0.30,
# ipe=0.25; pair with anvil friction mu=0.2).
# Multiplicative finite-strain Johnson-Cook for NEML2 -- ADIABATIC thermal
# variant for the full CuH04_235.9 slug run (temperature-rise state/dT
# formulation: T* = dT/1038 == (T-300)/(1338-300), lagged one step).
#
# Kinematics: F = Fe Fp (multiplicative split, Simo-style return map).
#   Trial state (Fp frozen at Fp_n): Fe_tr = F Fp_n^-1, Ee_tr = GL(Fe_tr),
#   S_tr = C : Ee_tr, N = dev(S_tr)-direction.
#   Radial return (1 unknown, state/ep): Ee = Ee_tr - dEp, S = C : Ee,
#   JC consistency via the inverted flow rate -- structurally IDENTICAL to
#   the verified additive chain, just driven by the trial ELASTIC strain.
#   Post-solve: Fp = (I + dEp) Fp_n (linearized exponential map,
#   det drift O(dep^2) per step -- negligible at explicit dt).
# Stress out: total PK2 = Fp^-1 S Fp^-T ('neml2_stress') and PK1 = F * PK2
#   ('pk1').
#
# Unlike the additive Green-Lagrange chain (johnson_cook_neml2_large_def.i),
# this stays valid under severe compression: the St.Venant-Kirchhoff
# pathology only bites at large ELASTIC strain, and here Fp carries the
# crush while Ee stays ~1e-3.
#
# state/Fp must be seeded with the identity:
#  - nodal-force path: [NEML2] identity_seeded_state = 'state/Fp'
#  - conventional path: initialize_outputs = 'state/Fp' + an identity
#    GenericConstantRankTwoTensor material.
#
# Full-hard OFHC copper (E = 117 GPa, nu = 0.34; JC A = 99.7 MPa,
# B = 262.8 MPa, n = 0.30 # calibrated, C = 0.029; initial_plastic_strain = 0 here to
# match the other test-scale models).

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    # elements sitting exactly on the elastic-plastic boundary oscillate around
    # the JC exponential kink at tight tolerances; 1e-7 on the ep-residual
    # (strain units) is far below any physical increment (~1e-3).
    abs_tol = 1e-7
    rel_tol = 1e-8
    max_its = 250
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  # ----- trial state: Fp frozen at Fp_n -----
  [trial_Fe]
    type = R2Multiplication
    A = 'deformation_gradient'
    B = 'state/Fp~1'
    invert_B = true
    to = 'forces/Fe_tr'
  []
  [trial_Ee]
    type = GreenLagrangeStrain
    deformation_gradient = 'forces/Fe_tr'
    strain = 'forces/Ee_tr'
  []
  [trial_S]
    type = LinearIsotropicElasticity
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '117e9 0.34' # OFHC copper
    strain = 'forces/Ee_tr'
    stress = 'forces/S_tr'
  []
  [flow_direction]
    type = AssociativeJ2FlowDirection
    mandel_stress = 'forces/S_tr'
    flow_direction = 'forces/N'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_Fe trial_Ee trial_S flow_direction'
    additional_outputs = 'forces/Ee_tr'
  []

  # ----- radial return: 1 unknown (state/ep), direction N fixed -----
  [ep_rate]
    type = ScalarVariableRate
    variable = 'state/ep'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/N'
    flow_rate = 'state/ep_rate'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/Ep'
  []
  [plastic_update]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate plastic_strain'
  []
  # Ee = Ee_tr - (Ep - Ep_n): only THIS step's plastic increment corrects the
  # trial elastic strain; prior plasticity already lives in Fp_n.
  [elastic_strain]
    type = SR2LinearCombination
    to = 'state/Ee'
    from = 'forces/Ee_tr state/Ep state/Ep~1'
    weights = '1 -1 1'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '117e9 0.34' # OFHC copper
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [stress_update]
    type = ComposedModel
    models = 'elastic_strain cauchy_stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [jc_flowrate]
    type = JohnsonCookFlowRate
    vonmises_stress = 'state/s'
    equivalent_plastic_strain = 'state/ep'
    use_temperature = true
    temperature = 'state/dT~1' # lagged one step (NEML2 history notation)
    flow_rate = 'state/ep_rate'
    A = 103799730.939 # calibrated to CuH04_235.9 (0.9 x literature)
    B = 328468110.085 # calibrated (0.9 x literature)
    n = 0.431173234185 # calibrated
    C = 0.02527548951
    m = 0.98
    reference_strain_rate = 1.0
    reference_temperature = 0
    melting_temperature = 1038
    initial_plastic_strain = 0.121266995145 # calibrated (H04 temper, fitted)
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/ep'
  []

  # adiabatic plastic heating: T_rate = beta/(rho*c_p) * sigma_vm * ep_dot
  #   beta = 0.9 (Taylor-Quinney), rho = 8960 kg/m^3, c_p = 385 J/(kg.K)  [OFHC Cu]
  #   scaling = 0.9 / (8960 * 385) = 2.6090e-7  K.m^3/J
  [plastic_heating]
    type = ScalarMultiplication
    from = 'state/s state/ep_rate'
    to = 'state/dT_rate'
    scaling = 2.6090e-7
  []
  [integrate_T]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/dT' # auto-uses state/dT_rate and state/dT~1
  []

  [rate]
    type = ComposedModel
    models = "plastic_update stress_update vonmises jc_flowrate integrate_ep"
  []
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'state/ep'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'return_map_sys'
    solver = 'newton'
    predictor = 'predictor'
  []

  # ----- post-solve: update Fp = (I + dEp) Fp_n -----
  [dEp]
    type = SR2LinearCombination
    from = 'state/Ep state/Ep~1'
    weights = '1 -1'
    to = 'state/dEp'
  []
  [dEp_r2]
    type = SR2ToR2
    input = 'state/dEp'
    output = 'state/dEp_r2'
  []
  [dEp_Fp]
    type = R2Multiplication
    A = 'state/dEp_r2'
    B = 'state/Fp~1'
    to = 'state/dEpFp'
  []
  [update_Fp]
    type = R2LinearCombination
    from = 'state/Fp~1 state/dEpFp'
    weights = '1 1'
    to = 'state/Fp'
  []

  # ----- post-solve stress measures from the converged state -----
  [pk2_r2]
    type = SR2ToR2
    input = 'state/S'
    output = 'state/S_r2'
  []
  [Fpinv_S]
    type = R2Multiplication
    A = 'state/Fp'
    B = 'state/S_r2'
    invert_A = true
    to = 'state/Fpinv_S'
  []
  [pk2_total]
    type = R2Multiplication
    A = 'state/Fpinv_S'
    B = 'state/Fp'
    invert_B = true
    transpose_B = true
    to = 'neml2_stress'
  []
  [pk1]
    type = R2Multiplication
    A = 'deformation_gradient'
    B = 'neml2_stress'
    to = 'pk1'
  []
  [model]
    type = ComposedModel
    models = 'trial_state return_map ep_rate plastic_update stress_update vonmises
              plastic_heating integrate_T
              dEp dEp_r2 dEp_Fp update_Fp pk2_r2 Fpinv_S pk2_total pk1'
    additional_outputs = 'state/s state/ep state/S state/Ep state/Fp state/dT neml2_stress'
  []
[]

[EquationSystems]
  [return_map_sys]
    type = NonlinearSystem
    model = 'rate'
    unknowns = 'state/ep'
  []
[]
