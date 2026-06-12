#
# 3D thermal dendritic solidification of a single, axis-aligned seed.
#
# Parameters are chosen so the dendrite tip operating point is resolvable on the
# grid (moderate undercooling), with interface-localized noise to trigger
# sidebranching. Expect fourfold arms along the coordinate axes with sidebranches
# developing at later times.
#

tau=1
W=1
D=2
# Thin-interface relation (Karma & Rappel): lambda = D*tau/(a2*W^2) with a2 = 0.6267
# yields a vanishing kinetic coefficient; larger lambda makes it negative (unstable).
a2=0.6267
lambda=${fparse D*tau/(a2*W*W)}
L=1
# Moderate undercooling: at |u_inf| ~ 0.8 the selected tip radius approaches the
# capillary length (sub-grid) and growth degenerates to compact octahedra.
u_inf=-0.55
# Anisotropy strength. eps_a = 4*eps4, so 0.2 corresponds to the standard eps4 = 0.05.
eps_a=0.2
# Small regularization to avoid division by zero at interface cores
eps_n=1e-8
r0=6
# Interface-localized noise amplitude; sidebranches are noise-amplified, a
# noiseless run produces smooth needles. Tune 0.01-0.05: more noise gives earlier
# sidebranching, too much roughens the tip.
noise_amp=0.02

[Domain]
  dim = 3
  nx = 256
  ny = 256
  nz = 256
  xmin = 0
  ymin = 0
  zmin = 0
  xmax = 256
  ymax = 256
  zmax = 256
  mesh_mode = DUMMY
  device_names = 'cuda'
[]

[GlobalParams]
  constant_names = 'tau W D lambda L u_inf eps_a eps_n r0 noise_amp'
  constant_expressions = '${tau} ${W} ${D} ${lambda} ${L} ${u_inf} ${eps_a} ${eps_n} ${r0} ${noise_amp}'
[]

[TensorBuffers]
  [rot]
    value_dimensions = '3 3'
  []
  [gradvec]
    value_dimensions = '3'
  []
  [fluxvec]
    value_dimensions = '3'
  []
[]

[TensorComputes]
  [Initialize]
    [phi]
      type = ParsedCompute
      buffer = phi
      extra_symbols = true
      expression = 'r := (x-128)^2+(y-128)^2+(z-128)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []

    [u]
      type = ParsedCompute
      buffer = u
      expression = 'u_inf'
      expand = REAL
    []

    # identity rotation: arms grow along the coordinate axes (swap in
    # RandomRotationTensor for a randomly oriented crystal)
    [rot]
      type = RankTwoIdentity
      buffer = rot
    []

    # linear reciprocal factors
    [phi_lin]
      type = ReciprocalLaplacianFactor
      buffer = phi_lin
      factor = ${fparse W*W/tau}
    []
    [u_lin]
      type = ReciprocalLaplacianFactor
      buffer = u_lin
      factor = ${D}
    []
  []

  [Solve]
    # anisotropic gradient energy flux
    [gradvec]
      type = GradientVector
      buffer = gradvec
      input = phi
    []
    [fluxvec]
      type = CubicAnisotropyFlux
      buffer = fluxvec
      gradient = gradvec
      rotation = rot
      eps_a = ${eps_a}
      eps_n = ${eps_n}
    []
    [anis]
      type = DivergenceVector
      buffer = anis
      input = fluxvec
      factor = ${fparse W*W}
    []

    # fresh noise every substep (no seed - a fixed seed would reproduce the same
    # field each evaluation). Generated on the device; cross-device
    # reproducibility is irrelevant for a noise source.
    [noise]
      type = RandomTensor
      buffer = noise
      min = -0.5
      max = 0.5
      generate_on_cpu = false
    []

    # explicit terms
    #
    # The driving force uses the quintic interpolant g'(phi) = 30*phi^2*(1-phi)^2, whose
    # derivative vanishes at phi=0,1. This keeps the bulk phases metastable for arbitrary
    # lambda*u. The noise is confined to the interface by the phi^2*(1-phi)^2 envelope
    # (peak value 1 at phi=0.5), so the bulk phases stay clean.
    [Rphi]
      type = ParsedCompute
      buffer = Rphi
      expression = 'fdw:=2*phi*(1-phi)*(1-2*phi); gp:=30*phi^2*(1-phi)^2; eta:=noise_amp*16*phi^2*(1-phi)^2*noise; (-fdw-lambda*u*gp+anis+eta)/tau'
      inputs = 'phi u anis noise'
    []

    [Rphibar]
      type = ForwardFFT
      buffer = Rphibar
      input = Rphi
    []
    [phibar]
      type = ForwardFFT
      buffer = phibar
      input = phi
    []

    # solid fraction, using the same quintic interpolant h(phi) = phi^3*(10-15*phi+6*phi^2)
    # as the driving force for thermodynamic consistency
    [s]
      type = ParsedCompute
      buffer = s
      expression = 'phi^3*(10-15*phi+6*phi^2)'
      inputs = 'phi'
    []

    # latent heat term
    [latent]
      type = LatentHeatSource
      buffer = latent
      s = s
      L = ${L}
    []
    [Nubar]
      type = ForwardFFT
      buffer = Nubar
      input = latent
    []
    [Nubar_fixed]
      type = ReciprocalMeanFix
      buffer = Nubar_fixed
      input = Nubar
      u_inf = 0.0
    []

    # temperature reciprocal buffer and mean fix (fixed far-field undercooling)
    [ubar]
      type = ForwardFFT
      buffer = ubar
      input = u
    []
    [ubar_fixed]
      type = ReciprocalMeanFix
      buffer = ubar_fixed
      input = ubar
      u_inf = ${u_inf}
    []
  []
[]

# The MOOSE timestep is the output cadence; the spectral integration runs on
# substeps (sub_dt = dt/substeps <= 0.1 at dtmax).
[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'phi u'
  reciprocal_buffer = 'phibar ubar_fixed'
  linear_reciprocal = 'phi_lin u_lin'
  nonlinear_reciprocal = 'Rphibar Nubar_fixed'
  substeps = 50
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 50
[]

[Executioner]
  type = Transient
  # ~120 steps reach t ~ 550; arms should span most of the box by then
  num_steps = 120
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.2
    dt = 0.05
  []
  dtmax = 5
[]

# NOTE: one frame is written per MOOSE timestep; at 256^3 in double precision
# that is ~134 MB per field per frame (~32 GB total for phi+u over 120 steps).
# Trim the buffer list or reduce num_steps to save disk.
[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'phi u'
    output_mode = 'Node Node'
    enable_hdf5 = true
  []
[]

[Outputs]
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
