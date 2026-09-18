interface_width = 2.0
r0 = 3

gbe_max = '${units 0.8797226 J/m^2}'

L = '${fparse 1.0 * 1.6 / ${interface_width}}'

g_gamma0 = '${fparse sqrt(2) / 3}' # g(gamma=1.5)
f0_gamma0 = 0.1411
kappa = '${fparse gbe_max * interface_width * sqrt(f0_gamma0) / g_gamma0}'
mu = '${fparse gbe_max / (g_gamma0 * interface_width * sqrt(f0_gamma0))}'

g_to_gamma_func = '(1.0/(-3.0944 * g^8 -1.8169*g^6 + 10.323 * g^4 - 8.1819*g^2 + 2.0033))'
dgamma_dsigma = '(g * (24.7552*g^6 + 10.9014*g^4 - 41.292*g^2 + 16.3638) / (-3.0944*g^8 - 1.8169*g^6 + 10.323*g^4 - 8.1819*g^2 + 2.0033)^2 / sqrt(${kappa} * ${mu}))'

f0 = '((gr0^4/4 - gr0^2/2) + (gr1^4/4 - gr1^2/2) + gamma_01*gr0^2*gr1^2 + 0.25)'

[Domain]
  dim = 2
  nx = 32
  ny = 32
  xmax = 8
  ymax = 8
  mesh_mode = DUMMY
  floating_precision = SINGLE
[]

[TensorComputes]
  [Initialize]
    [gr0]
      type = ParsedCompute
      buffer = 'gr0'
      extra_symbols = 'true'
      expression = 'radius:=sqrt((x-${Domain/xmax}/2)^2+(y-${Domain/ymax}/2)^2);0.5 - 0.5*tanh(2*(radius-${r0})/${interface_width})'
    []
    [gr1]
      type = ParsedCompute
      buffer = 'gr1'
      extra_symbols = 'true'
      expression = 'radius:=sqrt((x-${Domain/xmax}/2)^2+(y-${Domain/ymax}/2)^2);0.5 + 0.5*tanh(2*(radius-${r0})/${interface_width})'
    []
    [L_kappa_laplacian]
      type = ReciprocalLaplacianFactor
      buffer = L_kappa_laplacian
      factor = ${fparse ${L} * ${kappa}}
    []
    [kappa_laplacian]
      type = ReciprocalLaplacianFactor
      buffer = kappa_laplacian
      factor = '${kappa}'
    []
    [smooth]
      type = DeAliasingTensor
      method = HOULI
      buffer = smooth
    []
  []
  [Solve]
    ## FFT
    [gr0_bar]
      type = ForwardFFT
      buffer = gr0_bar
      input = gr0
    []
    [gr1_bar]
      type = ForwardFFT
      buffer = gr1_bar
      input = gr1
    []

    # Gradients
    [grad_gr0]
      type = GradientVector
      buffer = grad_gr0
      input = gr0_bar
      input_is_reciprocal = true
    []
    [grad_gr1]
      type = GradientVector
      buffer = grad_gr1
      input = gr1_bar
      input_is_reciprocal = true
    []

    # Get interface energy
    [sigma_gr0_gr1]
      type = PairwiseAnisotropicGBEnergy
      buffer = 'sigma_gr0_gr1'
      dsigma_dgrad_grain1 = 'dsigma_gr0_gr1_dgrad_gr0'
      dsigma_dgrad_grain2 = 'dsigma_gr0_gr1_dgrad_gr1'
      grad_grain1_buffer = 'grad_gr0'
      grad_grain2_buffer = 'grad_gr1'
      interface_width = '${interface_width}'
      libtorch_model_file = 'marlin:anisotropic_gb_energy/gb_energy_hull_2d.pt'
      chunk_size = 256
    []
    [gamma_01]
      type = ParsedCompute
      buffer = 'gamma_01'
      expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${g_to_gamma_func}'
      inputs = 'sigma_gr0_gr1'
    []
    [coeff_01]
      type = ParsedCompute
      buffer = 'coeff_01'
      expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr1^2'
      inputs = 'sigma_gr0_gr1 gr0 gr1'
    []
    [torque_01_gr0]
      type = AnisotropyTorque
      buffer = torque_01_gr0
      dmu_dn = dsigma_gr0_gr1_dgrad_gr0
      g = 'coeff_01'
    []
    [torque_01_gr1]
      type = AnisotropyTorque
      buffer = torque_01_gr1
      dmu_dn = dsigma_gr0_gr1_dgrad_gr1
      g = 'coeff_01'
    []
    [kappa_laplacian_gr0_hat]
      type = ParsedCompute
      buffer = kappa_laplacian_gr0_hat
      inputs = 'gr0_bar kappa_laplacian'
      expression = 'gr0_bar * kappa_laplacian'
    []
    [kappa_laplacian_gr1_hat]
      type = ParsedCompute
      buffer = kappa_laplacian_gr1_hat
      inputs = 'gr1_bar kappa_laplacian'
      expression = 'gr1_bar * kappa_laplacian'
    []

    [kappa_laplacian_gr0]
      type = InverseFFT
      buffer = kappa_laplacian_gr0
      input = kappa_laplacian_gr0_hat
    []
    [kappa_laplacian_gr1]
      type = InverseFFT
      buffer = kappa_laplacian_gr1
      input = kappa_laplacian_gr1_hat
    []

    [gr0_bulk_term]
      type = ParsedCompute
      buffer = 'gr0_bulk_term'
      expression = 'L_NL:=${L}*(sigma_gr0_gr1)/${gbe_max};-L_NL*${mu} * (gr0^3 - gr0 + 2*gr0*(gamma_01*gr1^2) - torque_01_gr0) + (L_NL - ${L})*kappa_laplacian_gr0'
      inputs = 'gr0 gr1 gamma_01 torque_01_gr0 sigma_gr0_gr1 kappa_laplacian_gr0'
    []
    [gr1_bulk_term]
      type = ParsedCompute
      buffer = 'gr1_bulk_term'
      expression = 'L_NL:=${L}*(sigma_gr0_gr1)/${gbe_max};-L_NL*${mu} * (gr1^3 - gr1 + 2*gr1*(gamma_01*gr0^2) - torque_01_gr1) + (L_NL - ${L})*kappa_laplacian_gr1'
      inputs = 'gr0 gr1 gamma_01 torque_01_gr1 sigma_gr0_gr1 kappa_laplacian_gr1'
    []

    ## build non-linear terms and smooth them
    [NL_gr0]
      type = ForwardFFT
      buffer = NL_gr0
      input = 'gr0_bulk_term'
    []
    [NL_gr1]
      type = ForwardFFT
      buffer = NL_gr1
      input = 'gr1_bulk_term'
    []
    [NL_gr0_smooth]
      type = ParsedCompute
      buffer = NL_gr0_smooth
      expression = 'smooth*NL_gr0'
      inputs = 'smooth NL_gr0'
    []
    [NL_gr1_smooth]
      type = ParsedCompute
      buffer = NL_gr1_smooth
      expression = 'smooth*NL_gr1'
      inputs = 'smooth NL_gr1'
    []

    # Properties for output
    [sigma]
      type = ParsedCompute
      buffer = 'sigma'
      expression = 'gr0^2*gr1^2*sigma_gr0_gr1'
      inputs = 'sigma_gr0_gr1 gr0 gr1'
    []

    [gradient_energy]
      type = GradientEnergyDensity
      buffer = gradient_energy
      gradient_buffers = 'grad_gr0 grad_gr1'
      kappa = ${kappa}
    []

    [total_energy]
      type = ParsedCompute
      buffer = 'total_energy'
      expression = '${mu} * (${f0}) + gradient_energy'
      inputs = 'gr0 gr1 gamma_01 gradient_energy'
    []
  []
[]

[Postprocessors]
  [total_gb_energy]
    type = TensorIntegralPostprocessor
    buffer = total_energy
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'gr0 gr1'
  reciprocal_buffer = 'gr0_bar gr1_bar'
  linear_reciprocal = 'L_kappa_laplacian L_kappa_laplacian'
  nonlinear_reciprocal = 'NL_gr0_smooth NL_gr1_smooth'
  substeps = 1e2
  predictor_order = 1
  corrector_order = 1
  corrector_steps = 1
[]

[TensorOutputs]
  active = ''
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'gr0 gr1 sigma_gr0_gr1 sigma total_energy'
    output_mode = 'NODE NODE NODE NODE NODE'
    enable_hdf5 = true
    transpose = false
  []
[]

[Executioner]
  type = Transient
  dt = 1
  num_steps = 2
[]

[Outputs]
  csv = true
  execute_on = 'INITIAL TIMESTEP_END'
[]
