#
# 3D thermal multigrain dendritic solidification with anisotropy
#

tau=1
W=1
D=2
lambda=8
gamma=10
L=1
u_inf=-0.8
eps_a=0.6
eps_n=1e-3
r0=6

[Domain]
  dim = 3
  nx = 128
  ny = 128
  nz = 128
  xmin = 0
  ymin = 0
  zmin = 0
  xmax = 128
  ymax = 128
  zmax = 128
  mesh_mode = DUMMY
  device_names = 'cuda'
[]

[GlobalParams]
  constant_names = 'tau W D lambda gamma L u_inf eps_a eps_n r0'
  constant_expressions = '${tau} ${W} ${D} ${lambda} ${gamma} ${L} ${u_inf} ${eps_a} ${eps_n} ${r0}'
[]

[TensorBuffers]
  # rotation matrices (value dims 3x3)
  [rot1]
    value_dimensions = '3 3'
  []
  [rot2]
    value_dimensions = '3 3'
  []
  [rot3]
    value_dimensions = '3 3'
  []
  [rot4]
    value_dimensions = '3 3'
  []
  [rot5]
    value_dimensions = '3 3'
  []

  # gradient vectors for anisotropy (value dims 3)
  [gradvec1]
    value_dimensions = '3'
  []
  [gradvec2]
    value_dimensions = '3'
  []
  [gradvec3]
    value_dimensions = '3'
  []
  [gradvec4]
    value_dimensions = '3'
  []
  [gradvec5]
    value_dimensions = '3'
  []

  # flux vectors (value dims 3)
  [fluxvec1]
    value_dimensions = '3'
  []
  [fluxvec2]
    value_dimensions = '3'
  []
  [fluxvec3]
    value_dimensions = '3'
  []
  [fluxvec4]
    value_dimensions = '3'
  []
  [fluxvec5]
    value_dimensions = '3'
  []
[]

[TensorComputes]
  [Initialize]
    [phi1]
      type = ParsedCompute
      buffer = phi1
      extra_symbols = true
      expression = 'r := (x-32)^2+(y-32)^2+(z-32)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []
    [phi2]
      type = ParsedCompute
      buffer = phi2
      extra_symbols = true
      expression = 'r := (x-96)^2+(y-32)^2+(z-96)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []
    [phi3]
      type = ParsedCompute
      buffer = phi3
      extra_symbols = true
      expression = 'r := (x-32)^2+(y-96)^2+(z-96)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []
    [phi4]
      type = ParsedCompute
      buffer = phi4
      extra_symbols = true
      expression = 'r := (x-96)^2+(y-96)^2+(z-32)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []
    [phi5]
      type = ParsedCompute
      buffer = phi5
      extra_symbols = true
      expression = 'r := (x-64)^2+(y-64)^2+(z-64)^2; 0.5*(1-tanh((sqrt(r)-r0)/(sqrt(2)*W)))'
    []

    [u]
      type = ParsedCompute
      buffer = u
      expression = 'u_inf'
      expand = REAL
    []

    # per-grain random rotations
    [rot1]
      type = RandomRotationTensor
      buffer = rot1
      seed = 1001
    []
    [rot2]
      type = RandomRotationTensor
      buffer = rot2
      seed = 1002
    []
    [rot3]
      type = RandomRotationTensor
      buffer = rot3
      seed = 1003
    []
    [rot4]
      type = RandomRotationTensor
      buffer = rot4
      seed = 1004
    []
    [rot5]
      type = RandomRotationTensor
      buffer = rot5
      seed = 1005
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
    # gradients
    [gradvec1]
      type = GradientVector
      buffer = gradvec1
      input = phi1
    []
    [a2m1_1]
      type = ParsedCompute
      buffer = a2m1_1
      expression = 'q:=matvec(transpose(rot1),gradvec1); q2:=vecvec(q,q); q4:=vecvec(q*q,q*q); d:=sqrt(q2)+eps_n; a:=1+eps_a*(q4/(d*d*d*d)-0.6); a*a-1'
      inputs = 'rot1 gradvec1'
    []
    [fluxvec1]
      type = ScaleVector
      buffer = fluxvec1
      scalar = a2m1_1
      vector = gradvec1
    []
    [anis1]
      type = DivergenceVector
      buffer = anis1
      input = fluxvec1
      factor = ${fparse W*W}
    []

    [gradvec2]
      type = GradientVector
      buffer = gradvec2
      input = phi2
    []
    [a2m1_2]
      type = ParsedCompute
      buffer = a2m1_2
      expression = 'q:=matvec(transpose(rot2),gradvec2); q2:=vecvec(q,q); q4:=vecvec(q*q,q*q); d:=sqrt(q2)+eps_n; a:=1+eps_a*(q4/(d*d*d*d)-0.6); a*a-1'
      inputs = 'rot2 gradvec2'
    []
    [fluxvec2]
      type = ScaleVector
      buffer = fluxvec2
      scalar = a2m1_2
      vector = gradvec2
    []
    [anis2]
      type = DivergenceVector
      buffer = anis2
      input = fluxvec2
      factor = ${fparse W*W}
    []

    [gradvec3]
      type = GradientVector
      buffer = gradvec3
      input = phi3
    []
    [a2m1_3]
      type = ParsedCompute
      buffer = a2m1_3
      expression = 'q:=matvec(transpose(rot3),gradvec3); q2:=vecvec(q,q); q4:=vecvec(q*q,q*q); d:=sqrt(q2)+eps_n; a:=1+eps_a*(q4/(d*d*d*d)-0.6); a*a-1'
      inputs = 'rot3 gradvec3'
    []
    [fluxvec3]
      type = ScaleVector
      buffer = fluxvec3
      scalar = a2m1_3
      vector = gradvec3
    []
    [anis3]
      type = DivergenceVector
      buffer = anis3
      input = fluxvec3
      factor = ${fparse W*W}
    []

    [gradvec4]
      type = GradientVector
      buffer = gradvec4
      input = phi4
    []
    [a2m1_4]
      type = ParsedCompute
      buffer = a2m1_4
      expression = 'q:=matvec(transpose(rot4),gradvec4); q2:=vecvec(q,q); q4:=vecvec(q*q,q*q); d:=sqrt(q2)+eps_n; a:=1+eps_a*(q4/(d*d*d*d)-0.6); a*a-1'
      inputs = 'rot4 gradvec4'
    []
    [fluxvec4]
      type = ScaleVector
      buffer = fluxvec4
      scalar = a2m1_4
      vector = gradvec4
    []
    [anis4]
      type = DivergenceVector
      buffer = anis4
      input = fluxvec4
      factor = ${fparse W*W}
    []

    [gradvec5]
      type = GradientVector
      buffer = gradvec5
      input = phi5
    []
    [a2m1_5]
      type = ParsedCompute
      buffer = a2m1_5
      expression = 'q:=matvec(transpose(rot5),gradvec5); q2:=vecvec(q,q); q4:=vecvec(q*q,q*q); d:=sqrt(q2)+eps_n; a:=1+eps_a*(q4/(d*d*d*d)-0.6); a*a-1'
      inputs = 'rot5 gradvec5'
    []
    [fluxvec5]
      type = ScaleVector
      buffer = fluxvec5
      scalar = a2m1_5
      vector = gradvec5
    []
    [anis5]
      type = DivergenceVector
      buffer = anis5
      input = fluxvec5
      factor = ${fparse W*W}
    []

    # overlap helper
    [sum_phi_sq]
      type = ParsedCompute
      buffer = sum_phi_sq
      expression = 'phi1^2+phi2^2+phi3^2+phi4^2+phi5^2'
      inputs = 'phi1 phi2 phi3 phi4 phi5'
    []

    # explicit terms for each phi_i
    [Rphi1]
      type = ParsedCompute
      buffer = Rphi1
      expression = 'fdw:=2*phi1*(1-phi1)*(1-2*phi1); ov:=2*gamma*phi1*(sum_phi_sq-phi1*phi1); gp:=6*phi1*(1-phi1); (-fdw-ov-lambda*u*gp+anis1)/(tau*(1+a2m1_1))'
      inputs = 'phi1 sum_phi_sq u anis1 a2m1_1'
    []
    [Rphi2]
      type = ParsedCompute
      buffer = Rphi2
      expression = 'fdw:=2*phi2*(1-phi2)*(1-2*phi2); ov:=2*gamma*phi2*(sum_phi_sq-phi2*phi2); gp:=6*phi2*(1-phi2); (-fdw-ov-lambda*u*gp+anis2)/(tau*(1+a2m1_2))'
      inputs = 'phi2 sum_phi_sq u anis2 a2m1_2'
    []
    [Rphi3]
      type = ParsedCompute
      buffer = Rphi3
      expression = 'fdw:=2*phi3*(1-phi3)*(1-2*phi3); ov:=2*gamma*phi3*(sum_phi_sq-phi3*phi3); gp:=6*phi3*(1-phi3); (-fdw-ov-lambda*u*gp+anis3)/(tau*(1+a2m1_3))'
      inputs = 'phi3 sum_phi_sq u anis3 a2m1_3'
    []
    [Rphi4]
      type = ParsedCompute
      buffer = Rphi4
      expression = 'fdw:=2*phi4*(1-phi4)*(1-2*phi4); ov:=2*gamma*phi4*(sum_phi_sq-phi4*phi4); gp:=6*phi4*(1-phi4); (-fdw-ov-lambda*u*gp+anis4)/(tau*(1+a2m1_4))'
      inputs = 'phi4 sum_phi_sq u anis4 a2m1_4'
    []
    [Rphi5]
      type = ParsedCompute
      buffer = Rphi5
      expression = 'fdw:=2*phi5*(1-phi5)*(1-2*phi5); ov:=2*gamma*phi5*(sum_phi_sq-phi5*phi5); gp:=6*phi5*(1-phi5); (-fdw-ov-lambda*u*gp+anis5)/(tau*(1+a2m1_5))'
      inputs = 'phi5 sum_phi_sq u anis5 a2m1_5'
    []

    [Rphi1bar]
      type = ForwardFFT
      buffer = Rphi1bar
      input = Rphi1
    []
    [Rphi2bar]
      type = ForwardFFT
      buffer = Rphi2bar
      input = Rphi2
    []
    [Rphi3bar]
      type = ForwardFFT
      buffer = Rphi3bar
      input = Rphi3
    []
    [Rphi4bar]
      type = ForwardFFT
      buffer = Rphi4bar
      input = Rphi4
    []
    [Rphi5bar]
      type = ForwardFFT
      buffer = Rphi5bar
      input = Rphi5
    []

    [phi1bar]
      type = ForwardFFT
      buffer = phi1bar
      input = phi1
    []
    [phi2bar]
      type = ForwardFFT
      buffer = phi2bar
      input = phi2
    []
    [phi3bar]
      type = ForwardFFT
      buffer = phi3bar
      input = phi3
    []
    [phi4bar]
      type = ForwardFFT
      buffer = phi4bar
      input = phi4
    []
    [phi5bar]
      type = ForwardFFT
      buffer = phi5bar
      input = phi5
    []

    # solid fraction
    [s]
      type = ParsedCompute
      buffer = s
      expression = 'phi1^2*(3-2*phi1)+phi2^2*(3-2*phi2)+phi3^2*(3-2*phi3)+phi4^2*(3-2*phi4)+phi5^2*(3-2*phi5)'
      inputs = 'phi1 phi2 phi3 phi4 phi5'
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

    # temperature reciprocal buffer and mean fix
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

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'phi1 phi2 phi3 phi4 phi5 u'
  reciprocal_buffer = 'phi1bar phi2bar phi3bar phi4bar phi5bar ubar_fixed'
  linear_reciprocal = 'phi_lin phi_lin phi_lin phi_lin phi_lin u_lin'
  nonlinear_reciprocal = 'Rphi1bar Rphi2bar Rphi3bar Rphi4bar Rphi5bar Nubar_fixed'
  substeps = 20
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 20
[]

[Executioner]
  type = Transient
  num_steps = 200
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.1
    dt = 0.01
  []
  dtmax = 0.1
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'phi1 phi2 phi3 phi4 phi5 u s a2m1_1 anis1'
    output_mode = 'Node Node Node Node Node Node Node Node Node'
    enable_hdf5 = true
  []
[]

[Outputs]
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
