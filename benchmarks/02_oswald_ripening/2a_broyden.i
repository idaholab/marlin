[Domain]
  dim = 2
  nx = 200
  ny = 200
  xmax = 200
  ymax = 200

  device_names = 'cuda:0'

  mesh_mode = DOMAIN
[]

fchem='fa:=rho^2*(c-ca)^2;
fb:=rho^2*(cb-c)^2;
h:=n1^3*(6*n1^2-15*n1+10) +
   n2^3*(6*n2^2-15*n2+10) +
   n3^3*(6*n3^2-15*n3+10) +
   n4^3*(6*n4^2-15*n4+10);
g:=n1^2*(1-n1)^2 +
   n2^2*(1-n2)^2 +
   n3^2*(1-n3)^2 +
   n4^2*(1-n4)^2 +
alpha*(
n1^2*n2^2 + n1^2*n3^2 + n1^2*n4^2 +
n2^2*n1^2 + n2^2*n3^2 + n2^2*n4^2 +
n3^2*n1^2 + n3^2*n2^2 + n3^2*n4^2 +
n4^2*n1^2 + n4^2*n2^2 + n4^2*n3^2);
(fa*(1-h) + fb*h + w*g)'

nic = 'epsilon*(cos((0.01*idx)*x-4)*cos((0.007+0.01*idx)*y)
       +cos((0.11+0.01*idx)*x)*cos((0.11+0.01*idx)*y)
       +psi*(cos((0.046+0.001*idx)*x+(0.0405+0.001*idx)*y)
       *cos((0.031+0.001*idx)*x-(0.004+0.001*idx)*y))^2)^2'

cnames = 'rho     ca  cb  alpha w L M'
cvalues= 'sqrt(2) 0.3 0.7 5     1 5 5'

[TensorBuffers]
  # variables
  [c]
    map_to_aux_variable = c
  []
  [n1]
  []
  [n2]
  []
  [n3]
  []
  [n4]
  []

  [c_bar]
  []
  [n1_bar]
  []
  [n2_bar]
  []
  [n3_bar]
  []
  [n4_bar]
  []

  [mu_c]
    # map_to_aux_variable = mu
  []
  [mu_n1]
  []
  [mu_n2]
  []
  [mu_n3]
  []
  [mu_n4]
  []

  [mu_c_bar]
  []
  [mu_n1_bar]
  []
  [mu_n2_bar]
  []
  [mu_n3_bar]
  []
  [mu_n4_bar]
  []

  [Mbar_mu_c_bar]
  []

  # constant tensors
  [Lbar] # FFT(M*laplacian)
  []
  [MkappaL2bar] # FFT(-M*kappa*laplacian^2)
  []
  [kappaLbar] # FFT(L*kappa*laplacian)
  []

  # postprocessing
  [F]
  []
  [Fgrad_c]
  []
  [Fgrad_n1]
  []
  [Fgrad_n2]
  []
  [Fgrad_n3]
  []
  [Fgrad_n4]
  []
  [bnds]
    map_to_aux_variable = bnds
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'c0+epsilon*(cos(0.105*x)*cos(0.11*y)+(cos(0.13*x)*cos(0.087*y))^2+cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y))'
      constant_names = 'c0 epsilon'
      constant_expressions = '0.5 0.01'
    []
    [Lbar]
      type = ReciprocalLaplacianFactor
      # Mobility is pulled into the chemical potential below
      buffer = Lbar
    []
    [MkappaL2bar]
      type = ReciprocalLaplacianSquareFactor
      factor = -15 # -kappa_c*M
      buffer = MkappaL2bar
    []
    [kappaLbar]
      type = ReciprocalLaplacianFactor
      buffer = kappaLbar
      factor = 15 # kappa_ni*L
    []
    [n1]
      type = ParsedCompute
      buffer = n1
      expression = ${nic}
      extra_symbols = true
      constant_names       = 'idx epsilon psi'
      constant_expressions = '  1     0.1 1.5'
    []
    [n2]
      type = ParsedCompute
      buffer = n2
      expression = ${nic}
      extra_symbols = true
      constant_names       = 'idx epsilon psi'
      constant_expressions = '  2    0.1 1.5'
    []
    [n3]
      type = ParsedCompute
      buffer = n3
      expression = ${nic}
      extra_symbols = true
      constant_names       = 'idx epsilon psi'
      constant_expressions = '  3     0.1 1.5'
    []
    [n4]
      type = ParsedCompute
      buffer = n4
      expression = ${nic}
      extra_symbols = true
      constant_names       = 'idx epsilon psi'
      constant_expressions = '  4     0.1 1.5'
    []
  []

  [Solve]
    [mu_c]
      type = ParsedCompute
      buffer = mu_c
      expression = '${fchem}*M'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = c
      inputs = 'c n1 n2 n3 n4'
    []

    [mu_n1]
      type = ParsedCompute
      buffer = mu_n1
      expression = '${fchem}*(-L)'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n1
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n2]
      type = ParsedCompute
      buffer = mu_n2
      expression = '${fchem}*(-L)'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n2
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n3]
      type = ParsedCompute
      buffer = mu_n3
      expression = '${fchem}*(-L)'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n3
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n4]
      type = ParsedCompute
      buffer = mu_n4
      expression = '${fchem}*(-L)'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n4
      inputs = 'c n1 n2 n3 n4'
    []

    [mu_c_bar]
      type = ForwardFFT
      buffer = mu_c_bar
      input = mu_c
    []
    [mu_n1_bar]
      type = ForwardFFT
      buffer = mu_n1_bar
      input = mu_n1
    []
    [mu_n2_bar]
      type = ForwardFFT
      buffer = mu_n2_bar
      input = mu_n2
    []
    [mu_n3_bar]
      type = ForwardFFT
      buffer = mu_n3_bar
      input = mu_n3
    []
    [mu_n4_bar]
      type = ForwardFFT
      buffer = mu_n4_bar
      input = mu_n4
    []

    [Mbar_mu_c_bar]
      type = ParsedCompute
      buffer = Mbar_mu_c_bar
      expression = 'Lbar*mu_c_bar'
      inputs = 'Lbar mu_c_bar'
    []

    [c_bar]
      type = ForwardFFT
      buffer = c_bar
      input = c
    []
    [n1_bar]
      type = ForwardFFT
      buffer = n1_bar
      input = n1
    []
    [n2_bar]
      type = ForwardFFT
      buffer = n2_bar
      input = n2
    []
    [n3_bar]
      type = ForwardFFT
      buffer = n3_bar
      input = n3
    []
    [n4_bar]
      type = ForwardFFT
      buffer = n4_bar
      input = n4
    []
  []

  [Postprocess]
    [Fgrad_c]
      type = FFTGradientSquare
      buffer = Fgrad_c
      input = c
      factor = 1.5 # kappa/2
    []
    [Fgrad_n1]
      type = FFTGradientSquare
      buffer = Fgrad_n1
      input = n1
      factor = 1.5 # kappa/2
    []
    [Fgrad_n2]
      type = FFTGradientSquare
      buffer = Fgrad_n2
      input = n2
      factor = 1.5 # kappa/2
    []
    [Fgrad_n3]
      type = FFTGradientSquare
      buffer = Fgrad_n3
      input = n3
      factor = 1.5 # kappa/2
    []
    [Fgrad_n4]
      type = FFTGradientSquare
      buffer = Fgrad_n4
      input = n4
      factor = 1.5 # kappa/2
    []
    [F]
      type = ParsedCompute
      buffer = F
      expression = '${fchem} + Fgrad_c + Fgrad_n1 + Fgrad_n2 + Fgrad_n3 + Fgrad_n4'
      constant_names =  ${cnames}
      constant_expressions = ${cvalues}
      inputs = 'c n1 n2 n3 n4 Fgrad_c Fgrad_n1 Fgrad_n2 Fgrad_n3 Fgrad_n4'
    []
    [bnds]
      type = ParsedCompute
      buffer = bnds
      expression = 'n1^2 + n2^2 + n3^2 + n4^2'
      inputs = 'n1 n2 n3 n4'
    []
  []
[]

[TensorSolver]
  type = BroydenSolver
  substeps = 10
  max_iterations = 1000
  damping = 0.5
  relative_tolerance = 1e-6
  absolute_tolerance = 1e-5
  buffer = 'c n1 n2 n3 n4'
  dt_epsilon = 1e-5
  reciprocal_buffer = 'c_bar n1_bar n2_bar n3_bar n4_bar'
  linear_reciprocal = 'MkappaL2bar kappaLbar kappaLbar kappaLbar kappaLbar'
  nonlinear_reciprocal = 'Mbar_mu_c_bar mu_n1_bar mu_n2_bar mu_n3_bar mu_n4_bar'
  # verbose = true
[]

[AuxVariables]
  [c]
  []
  [bnds]
  []
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [F]
    type = TensorIntegralPostprocessor
    buffer = F
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  # num_steps = 2
  end_time = 1e+5
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 1e-4
    max_iterations = 500
    min_iterations = 100
    growth_factor = 1.1
  []
[]

[Outputs]
  exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
