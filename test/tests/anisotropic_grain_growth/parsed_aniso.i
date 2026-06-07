interface_width = 0.8
r0 = 10

gbe_max = 1.0
kappa = '${fparse 0.75 * gbe_max * interface_width }'

grad_thresh = '2*(1/cosh(2*2))^2 / ${interface_width}'

sigma_a = '${units 1.0 J/m^2}'
sigma_b = '${units 0.1 J/m^2}'

g_eta_expr = '(eta^2*(1-eta^2)^2)'

L = 2.0

[Domain]
    dim = 2
    nx = 400
    ny = 400
    xmax = 40
    ymax = 40
    mesh_mode = DUMMY
    device_names = 'mps'
[]

[TensorComputes]
    [Initialize]
        [eta]
            type = ParsedCompute
            buffer = 'eta'
            extra_symbols = 'true'
            expression = 'radius:=sqrt((x-${Domain/xmax}/2)^2+(y-${Domain/ymax}/2)^2);0.5 - 0.5*tanh(2*(radius-${r0})/${interface_width})'
        []
        [kappa_linear_term]
            type = ReciprocalLaplacianFactor
            buffer = kappa_linear_term
            factor = '${fparse ${L} * ${kappa} }'
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
        [grad_x_eta]
            type = FFTGradient
            buffer = grad_x_eta
            input = eta
            direction = X
        []
        [grad_y_eta]
            type = FFTGradient
            buffer = grad_y_eta
            input = eta
            direction = Y
        []
        [grad_magnitude]
            type = ParsedCompute
            buffer = grad_magnitude
            inputs = 'grad_x_eta grad_y_eta'
            expression = 'sqrt(grad_x_eta^2 + grad_y_eta^2)'
        []
        [n_x]
            type = ParsedCompute
            buffer = 'n_x'
            inputs = 'grad_x_eta grad_magnitude'
            expression = 'if(grad_magnitude > ${grad_thresh},grad_x_eta/grad_magnitude,0.0)'
        []
        [n_y]
            type = ParsedCompute
            buffer = 'n_y'
            inputs = 'grad_y_eta grad_magnitude'
            expression = 'if(grad_magnitude > ${grad_thresh},grad_y_eta/grad_magnitude,0.0)'
        []
        [theta]
            type = ParsedCompute
            buffer = 'theta'
            inputs = 'grad_x_eta grad_y_eta grad_magnitude'
            expression = 'if(grad_magnitude > ${grad_thresh},atan(grad_y_eta/grad_x_eta),0.0)'
        []
        [sigma]
            type = ParsedCompute
            buffer = sigma
            inputs = 'theta'
            expression = 'sqrt((${sigma_a}*cos(theta))^2+(${sigma_b}*sin(theta))^2)'
        []

        [g_eta]
            type = ParsedCompute
            buffer = g_eta
            expression = '${g_eta_expr}'
            inputs = 'eta'
        []
        [dg_deta]
            type = ParsedCompute
            buffer = dg_deta
            expression = '${g_eta_expr}'
            inputs = 'eta'
            derivatives = 'eta'
        []
        [mu]
            type = ParsedCompute
            buffer = 'mu'
            expression = '6 * sigma / ${interface_width}'
            inputs = 'sigma'
        []

        [term1]
            type = ParsedCompute
            buffer = term1
            expression = '-${L} * dg_deta * mu'
            inputs = 'dg_deta mu'
        []

        [g_eta_dmu_dn_x]
            type = ParsedCompute
            buffer = g_eta_dmu_dn_x
            expression = 'if(grad_magnitude > ${grad_thresh}, g_eta * 6 * (${sigma_b}^2 - ${sigma_a}^2)*(grad_x_eta * grad_y_eta^2)/(${interface_width} * grad_magnitude^4 * sigma), 0.0)'
            inputs = 'g_eta grad_x_eta grad_y_eta grad_magnitude sigma'
        []
        [g_eta_dmu_dn_y]
            type = ParsedCompute
            buffer = g_eta_dmu_dn_y
            expression = 'if(grad_magnitude > ${grad_thresh}, -1.0 * g_eta * 6 * (${sigma_b}^2 - ${sigma_a}^2)*(grad_x_eta^2 * grad_y_eta)/(${interface_width} * grad_magnitude^4 * sigma), 0.0)'
            inputs = 'g_eta grad_x_eta grad_y_eta grad_magnitude sigma'
        []

        [grad_g_eta_dmu_dn_x]
            type = FFTGradient
            buffer = grad_g_eta_dmu_dn_x
            direction = X
            input = g_eta_dmu_dn_x
        []
        [grad_g_eta_dmu_dn_y]
            type = FFTGradient
            buffer = grad_g_eta_dmu_dn_y
            direction = Y
            input = g_eta_dmu_dn_y
        []

        [term2]
            type = ParsedCompute
            buffer = term2
            expression = '-${L} * (grad_g_eta_dmu_dn_x + grad_g_eta_dmu_dn_y)'
            inputs = 'grad_g_eta_dmu_dn_x grad_g_eta_dmu_dn_y'
        []

        [driving_force]
            type = ParsedCompute
            buffer = 'driving_force'
            expression = 'term1 + term2'
            inputs = 'term1 term2'
        []

        [etabar]
            type = ForwardFFT
            buffer = etabar
            input = eta
        []
        [NL]
            type = ForwardFFT
            buffer = NL
            input = term1 #driving_force
        []
        [NL_smooth]
            type = ParsedCompute
            buffer = NL_smooth
            expression = 'smooth * NL'
            inputs = 'smooth NL'
        []

        [kappa_laplacian_etabar]
            type = ParsedCompute
            buffer = kappa_laplacian_etabar
            expression = 'kappa_laplacian * etabar'
            inputs = 'kappa_laplacian etabar'
        []
        [kappa_laplacian_eta]
            type = InverseFFT
            buffer = kappa_laplacian_eta
            input = kappa_laplacian_etabar
        []

        [gb_energy]
            type = ParsedCompute
            buffer = gb_energy
            expression = 'g_eta * mu - kappa_laplacian_eta'
            inputs = 'g_eta mu kappa_laplacian_eta'
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'eta'
    reciprocal_buffer = 'etabar'
    linear_reciprocal = 'kappa_linear_term'
    nonlinear_reciprocal = 'NL_smooth'
    substeps = 1e2
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 2
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta sigma gb_energy term1 term2'
        output_mode = 'NODE NODE CELL CELL CELL'
        enable_hdf5 = true
        transpose = false
    []
[]

[Postprocessors]
    [total_gb_energy]
        type = TensorIntegralPostprocessor
        buffer = gb_energy
    []
[]

[Executioner]
    type = Transient
    dt = 0.2
    num_steps = 250
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]