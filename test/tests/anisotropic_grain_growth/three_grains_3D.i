interface_width = 1.6

gbe_max = '${units 1.3561 J/m^2}'
L = 1

left = '(0.5-0.5*tanh(2*(x-20)/${interface_width}))'
c1 = 'inside:=sqrt((x-16)^2+(y-20)^2+(z-20)^2)-7;(0.5-0.5*tanh(2*inside/${interface_width}))*${left}'
c2 = 'inside:=sqrt((x-24)^2+(y-20)^2+(z-20)^2)-7;(0.5-0.5*tanh(2*inside/${interface_width}))*(1-${left})'

g_gamma0 = '${fparse sqrt(2) / 3 }' # g(gamma=1.5)
f0_gamma0 = 0.1411
kappa = '${fparse gbe_max * interface_width * sqrt(f0_gamma0) / g_gamma0 }'
mu = '${fparse gbe_max  / (g_gamma0 * interface_width * sqrt(f0_gamma0) ) }'

g_to_gamma_func = '(1.0/(-3.0944 * g^8 -1.8169*g^6 + 10.323 * g^4 - 8.1819*g^2 + 2.0033))'
dgamma_dsigma = '(g * (24.7552*g^6 + 10.9014*g^4 - 41.292*g^2 + 16.3638) / (-3.0944*g^8 - 1.8169*g^6 + 10.323*g^4 - 8.1819*g^2 + 2.0033)^2 / sqrt(${kappa} * ${mu}) )'

f0 = '((gr0^4/4 - gr0^2/2) + (gr1^4/4 - gr1^2/2) + (gr2^4/4 - gr2^2/2) + gamma_01*gr0^2*gr1^2 + gamma_12*gr1^2*gr2^2 + gamma_02*gr0^2*gr2^2 + 0.25)'


[Domain]
    dim = 3
    nx = 100
    ny = 100
    nz = 100
    xmax = 40
    ymax = 40
    zmax = 40
    mesh_mode = DUMMY
    device_names = 'mps'
    floating_precision = SINGLE
[]

[TensorComputes]
    [Initialize]
        [gr0]
            type = ParsedCompute
            buffer = gr0
            expression = '${c1}'
            extra_symbols = true
        []
        [gr1]
            type = ParsedCompute
            buffer = gr1
            expression = '${c2}'
            extra_symbols = true
        []
        [gr2]
            type = ParsedCompute
            buffer = gr2
            expression = '1 - gr0 - gr1'
            inputs = 'gr0 gr1'
        []
        [L_kappa_laplacian]
            type = ReciprocalLaplacianFactor
            buffer = 'L_kappa_laplacian'
            factor = '${fparse ${L} * ${kappa}}'
        []
        [smooth]
            type = DeAliasingTensor
            method = HOULI
            buffer = smooth
        []
    []
    [Solve]
        [grad_gr0]
            type = GradientVector
            buffer = grad_gr0
            input = gr0
        []
        [grad_gr1]
            type = GradientVector
            buffer = grad_gr1
            input = gr1
        []
        [grad_gr2]
            type = GradientVector
            buffer = grad_gr2
            input = gr2
        []

        [sigma_gr0_gr1]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr1'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr1_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr1_dgrad_gr1'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr1'
            interface_width = '${interface_width}'
            libtorch_model_file = '/Users/bhavcv/projects/torch-gb5dof/R1_R2.pt'
        []
        [sigma_gr1_gr2]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr1_gr2'
            dsigma_dgrad_grain1 = 'dsigma_gr1_gr2_dgrad_gr1'
            dsigma_dgrad_grain2 = 'dsigma_gr1_gr2_dgrad_gr2'
            grad_grain1_buffer = 'grad_gr1'
            grad_grain2_buffer = 'grad_gr2'
            interface_width = '${interface_width}'
            libtorch_model_file = '/Users/bhavcv/projects/torch-gb5dof/R2_R3.pt'
        []
        [sigma_gr0_gr2]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr2'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr2_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr2_dgrad_gr2'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr2'
            interface_width = '${interface_width}'
            libtorch_model_file = '/Users/bhavcv/projects/torch-gb5dof/R1_R3.pt'
        []

        [gamma_01]
            type = ParsedCompute
            buffer = 'gamma_01'
            expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr1'
        []
        [gamma_12]
            type = ParsedCompute
            buffer = 'gamma_12'
            expression = 'g:=sigma_gr1_gr2/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr1_gr2'
        []
        [gamma_02]
            type = ParsedCompute
            buffer = 'gamma_02'
            expression = 'g:=sigma_gr0_gr2/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr2'
        []

        [coeff_01]
            type = ParsedCompute
            buffer = 'coeff_01'
            expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr1^2'
            inputs = 'sigma_gr0_gr1 gr0 gr1'
        []
        [coeff_12]
            type = ParsedCompute
            buffer = 'coeff_12'
            expression = 'g:=sigma_gr1_gr2/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr1^2 *gr2^2'
            inputs = 'sigma_gr1_gr2 gr1 gr2'
        []
        [coeff_02]
            type = ParsedCompute
            buffer = 'coeff_02'
            expression = 'g:=sigma_gr0_gr2/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr2^2'
            inputs = 'sigma_gr0_gr2 gr0 gr2'
        []

        [torque_01_gr0]
            type = AnisotropyTorque
            buffer = torque_01_gr0
            dmu_dn = dsigma_gr0_gr1_dgrad_gr0
            g = 'coeff_01'
            input = 'gr0'
        []
        [torque_01_gr1]
            type = AnisotropyTorque
            buffer = torque_01_gr1
            dmu_dn = dsigma_gr0_gr1_dgrad_gr1
            g = 'coeff_01'
            input = 'gr1'
        []
        [torque_02_gr0]
            type = AnisotropyTorque
            buffer = torque_02_gr0
            dmu_dn = dsigma_gr0_gr2_dgrad_gr0
            g = 'coeff_02'
            input = 'gr0'
        []
        [torque_02_gr2]
            type = AnisotropyTorque
            buffer = torque_02_gr2
            dmu_dn = dsigma_gr0_gr2_dgrad_gr2
            g = 'coeff_02'
            input = 'gr2'
        []
        [torque_12_gr1]
            type = AnisotropyTorque
            buffer = torque_12_gr1
            dmu_dn = dsigma_gr1_gr2_dgrad_gr1
            g = 'coeff_12'
            input = 'gr1'
        []
        [torque_12_gr2]
            type = AnisotropyTorque
            buffer = torque_12_gr2
            dmu_dn = dsigma_gr1_gr2_dgrad_gr2
            g = 'coeff_12'
            input = 'gr2'
        []

        [gr0_bulk_term]
            type = ParsedCompute
            buffer = 'gr0_bulk_term'
            expression = '-${L}*${mu} * (gr0^3 - gr0 + 2*gr0*(gamma_01*gr1^2 + gamma_02*gr2^2) - torque_01_gr0 - torque_02_gr0)'
            inputs = 'gr0 gr1 gr2 gamma_01 gamma_02 torque_01_gr0 torque_02_gr0'
        []
        [gr1_bulk_term]
            type = ParsedCompute
            buffer = 'gr1_bulk_term'
            expression = '-${L}*${mu} * (gr1^3 - gr1 + 2*gr1*(gamma_01*gr0^2 + gamma_12*gr2^2) - torque_01_gr1 - torque_12_gr1)'
            inputs = 'gr0 gr1 gr2 gamma_01 gamma_12 torque_01_gr1 torque_12_gr1'
        []
        [gr2_bulk_term]
            type = ParsedCompute
            buffer = 'gr2_bulk_term'
            expression = '-${L}*${mu} * (gr2^3 - gr2 + 2*gr2*(gamma_02*gr0^2 + gamma_12*gr1^2) - torque_02_gr2 - torque_12_gr2)'
            inputs = 'gr0 gr1 gr2 gamma_02 gamma_12 torque_02_gr2 torque_12_gr2'
        []

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
        [gr2_bar]
            type = ForwardFFT
            buffer = gr2_bar
            input = gr2
        []

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
        [NL_gr2]
            type = ForwardFFT
            buffer = NL_gr2
            input = 'gr2_bulk_term'
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
        [NL_gr2_smooth]
            type = ParsedCompute
            buffer = NL_gr2_smooth
            expression = 'smooth*NL_gr2'
            inputs = 'smooth NL_gr2'
        []

        # Properties for output
        [sigma]
            type = ParsedCompute
            buffer = 'sigma'
            expression = '(gr0^2*gr1^2*sigma_gr0_gr1 + gr1^2*gr2^2*sigma_gr1_gr2 + gr0^2*gr2^2*sigma_gr0_gr2)/(gr0^2*gr1^2 + gr1^2*gr2^2 + gr0^2*gr2^2 + 1e-3)'
            inputs = 'sigma_gr0_gr1 sigma_gr0_gr2 sigma_gr1_gr2 gr0 gr1 gr2'
        []
        [total_energy]
            type = ParsedCompute
            buffer = 'total_energy'
            expression = '${mu} * (${f0}) + kappa_laplacian_gr_i'
            inputs = 'gr0 gr1 gr2 gamma_01 gamma_12 gamma_02 kappa_laplacian_gr_i'
        []
        [kappa_laplacian_gr_i_bar]
            type = ParsedCompute
            buffer = kappa_laplacian_gr_i
            expression = '${kappa} * (k2*gr0_bar + k2*gr1_bar +k2*gr2_bar )'
            inputs = 'gr0_bar gr1_bar gr2_bar'
            extra_symbols = true
        []
        [kappa_laplacian_gr_i]
            type = InverseFFT
            buffer = kappa_laplacian_gr_i
            input = kappa_laplacian_gr_i_bar            
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
    buffer = 'gr0 gr1 gr2'
    reciprocal_buffer = 'gr0_bar gr1_bar gr2_bar'
    linear_reciprocal = 'L_kappa_laplacian L_kappa_laplacian L_kappa_laplacian'
    nonlinear_reciprocal = 'NL_gr0_smooth NL_gr1_smooth NL_gr2_smooth'
    substeps = 10
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'gr0 gr1 gr2 sigma_gr0_gr1 sigma_gr1_gr2 sigma_gr0_gr2 sigma'
        output_mode = 'NODE NODE NODE NODE NODE NODE NODE'
        enable_hdf5 = true
        transpose = true
    []
[]

[Executioner]
    type = Transient
    dt = 0.1
    num_steps = 50
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
