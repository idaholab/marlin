interface_width = 0.8
r0 = 10

gbe_max = ${units 0.8 J/m^2}
kappa = ${fparse 0.75 * gbe_max * interface_width }
L = 1

g_eta_expr = '(eta^2*(1-eta^2)^2)'

[Domain]
  dim = 2
  nx = 400
  ny = 400
  xmax = 40
  ymax = 40
  mesh_mode = DUMMY
  device_names = 'mps'
  floating_precision = SINGLE
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
            factor = ${fparse ${L} * ${kappa} }
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
        [grad_eta_vector]
            type = GradientVector
            buffer = grad_eta
            input = eta
        []
        [gb_energy]
            type = AnisotropicGBEnergy
            buffer = gb_energy
            gb_gradient_buffer = grad_eta
            dsigma_dn = dsigma_dn
            libtorch_model_file = '/Users/bhavcv/projects/torch-gb5dof/gb_energy_hull_3d.pt'
            interface_width = ${interface_width}
        []
        [bulk_driving_force]
            type = ParsedCompute
            buffer = 'bulk_driving_force'
            expression = '6 * gb_energy * ${g_eta_expr} / ${interface_width}'
            inputs = 'mu eta gb_energy'
            derivatives = 'eta'
        []
        [dmu_dn]
            type = ParsedCompute
            buffer = 'dmu_dn'
            expression = '6 * dsigma_dn / ${interface_width}'
            inputs = 'dsigma_dn'
        []
        [g]
            type = ParsedCompute
            buffer = 'g'
            expression = '${g_eta_expr}'
            inputs = 'eta'
        []
        [anisotropy_torque]
            type = AnisotropyTorque
            buffer = anisotropy_torque
            input = eta
            dmu_dn = dmu_dn
            g = g
        []
        [total_mu_driving_force]
            type = ParsedCompute
            buffer = 'total_mu_driving_force'
            expression = '-${L}*(bulk_driving_force - anisotropy_torque)'
            inputs = 'bulk_driving_force anisotropy_torque'
        []

        [nonlinear]
            type = ForwardFFT
            buffer = 'NL'
            input = 'total_mu_driving_force'
        []
        [smooth_nonlinear]
            type = ParsedCompute
            buffer = 'smooth_NL'
            expression = 'smooth * NL'
            inputs = 'smooth NL'
        []

        [etabar]
            type = ForwardFFT
            buffer = etabar
            input = eta
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

        [interface_energy]
            type = ParsedCompute
            buffer = interface_energy
            expression = 'g * 6 * gb_energy / ${interface_width} - kappa_laplacian_eta'
            inputs = 'g gb_energy kappa_laplacian_eta'
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'eta'
    reciprocal_buffer = 'etabar'
    linear_reciprocal = 'kappa_linear_term'
    nonlinear_reciprocal = 'smooth_NL'
    substeps = 1e3
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[Postprocessors]
    [total_gb_energy]
        type = TensorIntegralPostprocessor
        buffer = interface_energy
    []
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta gb_energy'
        output_mode = 'NODE NODE'
        enable_hdf5 = true
        transpose = false
    []
[]

[Executioner]
    type = Transient
    dt = 0.1
    num_steps = 20
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
