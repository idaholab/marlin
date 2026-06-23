#### Compute ####
[Solve]

  # For temperature
  [Temperature]
    type = LBMComputeDensity
    buffer = T
    f = g
  []

  # For fluid
  [Fluid_density]
    type = LBMComputeDensity
    buffer = density
    f = f
  []

  [Fluid_velocity]
    type = LBMComputeVelocity
    buffer = velocity
    f = f
    rho = density
    forces = F
    enable_forces = true
  []

  # For temperature
  [Equilibrium_temperature]
    type = LBMEquilibrium
    buffer = geq
    bulk = T
    velocity = velocity
  []

  [Collision_temperature]
    type = LBMBGKCollision
    buffer = gpc
    f = g
    feq = geq
    tau0 = tau_T
  []

  # For fluid
  [Compute_forces]
    type = LBMComputeForces
    buffer = F
    rho0 = 'rho0'
    temperature = T
    T0 = T_0
    enable_buoyancy = true
    gravity = g
  []

  [Equilibrium_fluid]
    type = LBMEquilibrium
    buffer = feq
    bulk = density
    velocity = velocity
  []

  [Collision_fluid]
    type = LBMBGKCollision
    buffer = fpc
    f = f
    feq = feq
    tau0 = tau_f
  []

  [Apply_forces]
    type = LBMApplyForces
    buffer = fpc
    velocity = velocity
    rho = density
    forces = F
    tau0 = tau_f
  []
[]
