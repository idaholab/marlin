#
# Spinodal decomposition
# PHYSICAL REVIEW E 97, 033309 (2018) - Section III.C
#

# Domain
Nx = 200
Ny = 200

# Fluid properties
rho_l = 1000.0
rho_g = 1.0
nu_l = 0.1
nu_g = 1.0
sigma = 0.2

# Phase field parameters
tau_h = 0.67
D = 4

[Domain]
  dim = 2
  nx = '${Nx}'
  ny = '${Ny}'
  xmax = '${Nx}'
  ymax = '${Ny}'
  device_names='cpu'
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  # Macroscopic phase field variables
  [phi]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [grad_phi]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [laplacian_phi]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [mu]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [forces]
    type = LBMTensorBuffer
    buffer_type = mv
  []

  # Macroscopic hydrodynamic variables
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [pressure]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [rho]
    type = LBMTensorBuffer
    buffer_type = ms
  []

  # LBM phase field variabels
  [h]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [h_post_collision]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [h_eq]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [relaxation_tensor]
    type = LBMTensorBuffer
    buffer_type = ms
  []

  # LBM hydrodynamic variables
  [fdummy]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [f]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [f_post_collision]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [f_eq]
    type = LBMTensorBuffer
    buffer_type = df
  []
[]

[TensorComputes/Initialize]
  [phi_init]
    type = RandomTensor
    buffer = phi
    min = 0.3233
    max = 0.3433
  []
  [grad_phi_init]
    type = LBMIsotropicGradient
    buffer = grad_phi
    scalar_field = phi
  []
  [rho_init]
    type = ParsedCompute
    buffer = rho
    extra_symbols = true
    expression = 'phi*(rho_l - rho_g) + rho_g'
    constant_names = 'rho_l rho_g'
    constant_expressions = '${rho_l} ${rho_g}'
    inputs = phi
  []
  [pressure_init]
    type = LBMConstantTensor
    buffer = pressure
    constants = 0.3
  []
  # Phase field equilibrium distribution initialization
  [h_eq_init]
    type = LBMPhaseEquilibrium
    buffer = h_eq
    phi = phi
    velocity = velocity
  []
  [h_post_collision_init]
    type = LBMPhaseEquilibrium
    buffer = h_post_collision
    phi = phi
    velocity = velocity
  []
  [h_init]
    type = LBMPhaseEquilibrium
    buffer = h
    phi = phi
    velocity = velocity
  []
  # Hydrodynamic equilibrium distribution initialization
  [f_eq_init]
    type = LBMPressureCorrectedEquilibrium
    buffer = f_eq
    rho = rho
    velocity = velocity
    pressure = pressure
  []
  [f_post_collision_init]
    type = ParsedCompute
    buffer = f_post_collision
    expression = 'f_eq'
    inputs = f_eq
  []
  [f_init]
    type = ParsedCompute
    buffer = f
    expression = 'f_eq'
    inputs = f_eq
  []
[]

[TensorComputes/Solve]
  # Phase Field
  [compute_phi]
    type = LBMComputeDensity
    buffer = phi
    f = h
  []
  [grad_phi]
    type = LBMIsotropicGradient
    buffer = grad_phi
    scalar_field = phi
  []
  [laplacian_phi]
    type = LBMIsotropicLaplacian
    buffer = laplacian_phi
    scalar_field = phi
  []
  [potential]
    type = LBMComputeChemicalPotential
    buffer = mu
    phi = phi
    laplacian_phi = laplacian_phi
    thickness = D
    sigma = sigma
  []
  [forces]
    type = LBMComputeSurfaceForces
    buffer = forces
    chemical_potential = mu
    grad_phi = grad_phi
  []
  # Hydrodynamics
  [density]
    type = ParsedCompute
    buffer = rho
    extra_symbols = true
    expression = 'phi*(rho_l - rho_g) + rho_g'
    constant_names = 'rho_l rho_g'
    constant_expressions = '${rho_l} ${rho_g}'
    inputs = phi
  []
  [velocity]
    type = LBMComputeVelocity
    buffer = velocity
    f = f
    rho = rho
    enable_forces = true
    forces = forces
  []
  # Phase-field
  [h_eq]
    type = LBMPhaseEquilibrium
    buffer = h_eq
    phi = phi
    velocity = velocity
  []
  [phase_collision]
    type = LBMBGKCollision
    buffer = h_post_collision
    f = h
    feq = h_eq
    tau0 = tau_h
  []
  [apply_forces_phase]
    type = LBMAllenCahnSource
    buffer = h_post_collision
    phi = phi
    velocity = velocity
    grad_phi = grad_phi
    tau = tau_h
    thickness = D
  []
  # Hydrodynamics
  [relaxation_tensor]
    type = ParsedCompute
    buffer = relaxation_tensor
    extra_symbols = true
    expression = '(phi*(nu_l - nu_g) + nu_g)/cs2+0.5'
    constant_names = 'nu_l nu_g cs2'
    constant_expressions = '${nu_l} ${nu_g} 0.3333'
    inputs = phi
  []
  [pressure]
    type = LBMPhaseFieldPressure
    buffer = pressure
    f = f
    velocity = velocity
    grad_phi = grad_phi
    rho = rho
    rho_l = '${rho_l}'
    rho_g = '${rho_g}'
  []
  [f_eq]
    type = LBMPressureCorrectedEquilibrium
    buffer = f_eq
    rho = rho
    velocity = velocity
    pressure = pressure
  []
  [collision]
    type = LBMBGKCollision
    buffer = f_post_collision
    f = f
    feq = f_eq
    tau0 = 1.0
    is_dynamic_relaxation = true
    tau_tensor = relaxation_tensor
   []
  [apply_forces_hydro]
    type = LBMForceDistribution
    buffer = f_post_collision
    grad_phi = grad_phi
    velocity = velocity
    forces = forces
    tau_tensor = relaxation_tensor
    tau = 1.0
    rho_l = '${rho_l}'
    rho_g = '${rho_g}'
    is_dynamic_relaxation = true
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = 'h f'
  f_old = 'h_post_collision f_post_collision'
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 500
  print_debug_output = true
  scalar_constant_names = 'tau_h D sigma'
  scalar_constant_values = '${tau_h} ${D} ${sigma}'
  log_interval = 100
  residual_tensor = speed
[]

[Executioner]
  type = Transient
  num_steps = 100
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'phi rho velocity'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
    # transpose = false
  []
[]
