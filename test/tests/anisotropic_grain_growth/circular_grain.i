interface_width = 4
r0 = 6

gbe_max = ${units 0.1 J/m^2}
kappa = ${fparse 0.75 * gbe_max * interface_width }
L = 1

[Domain]
  dim = 2
  nx = 100
  ny = 100
  xmax = 40
  ymax = 40
  mesh_mode = DUMMY
#   device_names = 'mps'
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
            factor = ${kappa}
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
            gb_normal_buffer = grad_eta
            dsigma_dn = dsigma_dn
            libtorch_model_file = 'gb_energy_hull_3d.pt'
        []
        [mu]
            type = ParsedCompute
            buffer = 'mu'
            expression = '6 * gb_energy / ${interface_width}'
            inputs = 'gb_energy'
        []
        [bulk_driving_force]
            type = ParsedCompute
            buffer = 'bulk_driving_force'
            expression = 'mu * eta^2*(1-eta^2)^2'
            inputs = 'mu eta'
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
            expression = 'eta^2*(1-eta^2)^2'
            inputs = 'eta'
        []
        [anisotropy_torque]
            type = AnisotropyTorque
            buffer = anisotropy_torque
            input = eta
            dmu_dn = dmu_dn
            g = g
        []

        [fft_torque]
            type = ForwardFFT
            buffer = fft_torque
            input = anisotropy_torque
        []
        [smooth_fft_torque]
            type = ParsedCompute
            buffer = 'smooth_fft_torque'
            expression = 'smooth * fft_torque'
            inputs = 'smooth fft_torque'
        []
        [smooth_torque]
            type = InverseFFT
            buffer = smooth_torque
            input = smooth_fft_torque
        []

        [total_mu_driving_force]
            type = ParsedCompute
            buffer = 'total_mu_driving_force'
            expression = '-${L}*(bulk_driving_force)'# - anisotropy_torque)'
            inputs = 'bulk_driving_force'# anisotropy_torque'
        []

        [nonlinear]
            type = ForwardFFT
            buffer = 'NL'
            input = 'bulk_driving_force' #'total_mu_driving_force'
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
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'eta'
    reciprocal_buffer = 'etabar'
    linear_reciprocal = 'kappa_linear_term'
    nonlinear_reciprocal = 'NL'
    substeps = 1e3
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta grad_eta gb_energy dsigma_dn mu g anisotropy_torque total_mu_driving_force smooth_torque'
        enable_hdf5 = true
        transpose = false
    []
[]

[Executioner]
    type = Transient
    dt = 0.1
    num_steps = 1000
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
