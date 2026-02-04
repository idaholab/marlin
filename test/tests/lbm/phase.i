[Domain]
  dim = 2
  nx = 8
  ny = 8
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
    file = phi.h5
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
[]

[TensorComputes]
  [Initialize]
     [grad_phi_init]
      type = LBMIsotropicGradient
      buffer = grad_phi
      scalar_field = phi
    []
    [laplacian_phi_init]
      type = LBMIsotropicLaplacian
      buffer = laplacian_phi
      scalar_field = phi
    []
    [potential_init]
      type = LBMComputeChemicalPotential
      buffer = mu
      phi = phi
      laplacian_phi = laplacian_phi
      thickness = D
      sigma = sigma
    []
    [h_init]
      type = LBMPhaseEquilibrium
      buffer = h
      phi = phi
      grad_phi = grad_phi
      tau_phi = tau
      thickness = D
    []
    [h_init_pc]
      type = LBMPhaseEquilibrium
      buffer = h_post_collision
      phi = phi
      grad_phi = grad_phi
      tau_phi = tau
      thickness = D
    []
  []

  [Solve]
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
    [h_eq]
      type = LBMPhaseEquilibrium
      buffer = h_eq
      phi = phi
      grad_phi = grad_phi
      tau_phi = tau
      thickness = D
    []
    [phase_collision]
      type = LBMBGKCollision
      buffer = h_post_collision
      f = h
      feq = h_eq
      tau0 = tau
    []
    [apply_forces]
      type = LBMApplyForces
      buffer = h_post_collision
      rho = phi
      forces = forces
      tau0 = tau
    []
    [residual]
      type = LBMComputeResidual
      buffer = phi
      speed = phi
    []
  []
[]

[TensorSolver]
  type = LBMStream
  root_compute=residual
  buffer = h
  f_old = h_post_collision
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
  print_debug_output = true
  scalar_constant_names = 'tau D sigma'
  scalar_constant_values = '2.0 1 0.01'
[]

[Postprocessors]
  [phi_min]
    type = TensorExtremeValuePostprocessor
    buffer = phi
    value_type = MIN
  []
  [phi_max]
    type = TensorExtremeValuePostprocessor
    buffer = phi
    value_type = MAX
  []
  [grad_phi_min]
    type = TensorExtremeValuePostprocessor
    buffer = grad_phi
    value_type = MIN
  []
  [grad_phi_max]
    type = TensorExtremeValuePostprocessor
    buffer = grad_phi
    value_type = MAX
  []
  [laplacian_min]
    type = TensorExtremeValuePostprocessor
    buffer = laplacian_phi
    value_type = MIN
  []
  [laplacian_max]
    type = TensorExtremeValuePostprocessor
    buffer = laplacian_phi
    value_type = MAX
  []
  [mu_min]
    type = TensorExtremeValuePostprocessor
    buffer = mu
    value_type = MIN
  []
  [mu_max]
    type = TensorExtremeValuePostprocessor
    buffer = mu
    value_type = MAX
  []
  [forces_min]
    type = TensorExtremeValuePostprocessor
    buffer = forces
    value_type = MIN
  []
  [forces_max]
    type = TensorExtremeValuePostprocessor
    buffer = forces
    value_type = MAX
  []
[]

[Executioner]
  type = Transient
  num_steps = 10
[]

[Outputs]
  file_base = phase
  csv = true
[]
