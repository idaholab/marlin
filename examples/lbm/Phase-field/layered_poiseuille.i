#
# Layered Poiseuille Flow
# PHYSICAL REVIEW E 97, 033309 (2018) - Section III.B
#

# Domain
Nx = 10
Ny = 400

# Fluid properties
rho_l = 1000.0
rho_g = 1.0
# nu_l = 0.1
# nu_g = 1.0
sigma = 0.001
mu_l = 100.0 # rho_l * nu_l
mu_g = 1.0   # rho_g * nu_g

# Phase field parameters
# M = 0.1
# cs2 = 0.333333333333
tau_h = 0.8 # 0.5 + '${M}' / '${cs2}'
D = 5

# Driving force: Gx = uc * (mu_l + mu_g) / h^2
# uc = 1e-4
# h = # '${Ny} / 2'
Gx = 2.53e-07 #  '${uc} * (${mu_l}  + ${mu_g}) / (${h}^2)'

[Domain]
  dim = 2
  nx = '${Nx}'
  ny = '${Ny}'
  xmax = '${Nx}'
  ymax = '${Ny}'
  device_names = 'cpu'
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
    file = phi_init.h5
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
  [body_force]
    type = LBMTensorBuffer
    buffer_type = mv
  []

  # Macroscopic hydrodynamic variables
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  [speed]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [pressure]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [rho]
    type = LBMTensorBuffer
    buffer_type = ms
  []

  # LBM phase field variables
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
  [grad_phi_init]
    type = LBMIsotropicGradient
    buffer = grad_phi
    scalar_field = phi
  []
  [rho_init]
    type = ParsedCompute
    buffer = rho
    expression = 'phi * (rho_l - rho_g) + rho_g'
    constant_names = 'rho_l rho_g'
    constant_expressions = '${rho_l} ${rho_g}'
    inputs = phi
  []
  [pressure_init]
    type = LBMConstantTensor
    buffer = pressure
    constants = 0.3
  []
  [body_force_init]
    type = LBMConstantTensor
    buffer = body_force
    constants = '${Gx} 0.00'
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
  [compute_forces]
    type = LBMComputeSurfaceForces
    buffer = forces
    chemical_potential = mu
    grad_phi = grad_phi
  []
  [add_body_force]
    type = ParsedCompute
    buffer = forces
    expression = 'forces + body_force'
    inputs = 'forces body_force'
  []
  # Hydrodynamics
  [density]
    type = ParsedCompute
    buffer = rho
    expression = 'phi * (rho_l - rho_g) + rho_g'
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
    # add_body_force = true
    # body_force_x = '${Gx}'
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
    # Implements Eq 26: Sharp step-function for dynamic viscosity
    expression = '(if(phi >= 0.5, mu_l, mu_g) / rho) / cs2 + 0.5'
    constant_names = 'mu_l mu_g cs2'
    constant_expressions = '${mu_l} ${mu_g} 0.333333333333'
    inputs = 'phi rho'
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
  [speed]
    type = LBMComputeVelocityMagnitude
    buffer = speed
    velocity = velocity
  []
[]

[TensorComputes/Boundary]
  [top_fluid]
    type = LBMBounceBack
    buffer = f
    f_old = f_post_collision
    boundary = top
  []
  [bottom_fluid]
    type = LBMBounceBack
    buffer = f
    f_old = f_post_collision
    boundary = bottom
  []
  [top_phase]
    type = LBMBounceBack
    buffer = h
    f_old = h_post_collision
    boundary = top
  []
  [bottom_phase]
    type = LBMBounceBack
    buffer = h
    f_old = h_post_collision
    boundary = bottom
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = 'h f'
  f_old = 'h_post_collision f_post_collision'
[]

[Problem]
  type = LatticeBoltzmannProblem
  # Keep this low for this setup: high substeps with top/bottom bounce-back can blow up to NaN.
  substeps = 100000
  print_debug_output = true
  scalar_constant_names = 'tau_h D sigma'
  scalar_constant_values = '${tau_h} ${D} ${sigma}'
  residual_tensor = speed
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'phi velocity rho'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
    transpose = false
  []
[]
