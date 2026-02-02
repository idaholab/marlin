# FCC PFC test designed to clearly show atomic-scale density fluctuations
# Using larger epsilon for stronger density modulations
# Based on PhysRevE.81.061601 two-mode FCC PFC model

# Grid parameters - fine resolution to resolve atomic peaks
N = 4  # Grid points per dimension

# PFC model parameters - LARGER epsilon for clearer atomic structure
psi_mean = -0.2         # Mean density
epsilon = 0.15          # Larger undercooling for stronger modulations
R1_param = 0.0          # Two-mode coupling (0 for max FCC stability)
Q1_param = ${fparse sqrt(4.0/3.0)}  # FCC wave number ratio

# Domain size: want several unit cells but fine enough resolution
# The (111) wavelength is 2*pi/q0 = 2*pi*sqrt(3/4) ≈ 5.44 (dimensionless)
# Let's fit about 'N' wavelengths in each direction
Lx = ${fparse N * 2 * pi / ${Q1_param}}
Ly = ${Lx}

[Domain]
    dim = 2
    nx = ${fparse ${N} * 8 }
    ny = ${fparse ${N} * 8 }
    xmax = ${Lx}
    ymax = ${Ly}
[]

[TensorComputes]
    [Initialize]
        # Random perturbations to seed crystal growth
        [psi]
            type = RandomTensor
            buffer = psi
            max = ${fparse ${psi_mean} + 0.01}
            min = ${fparse ${psi_mean} - 0.01}
            seed = 12345
        []

        # Linear operator for FCC
        [linear]
            type = FCCPFCLinear
            buffer = 'linear'
            eps = ${epsilon}
            R1 = ${R1_param}
            Q1 = ${Q1_param}
            mobility = 1.0
        []

        # Dealiasing for cubic nonlinearity
        [smooth_operator]
            type = DeAliasingTensor
            buffer = smooth_operator
            method = HOULI
        []
    []

    [Solve]
        # Nonlinear term
        [nl_div_psi_cubed]
            type = FCCPFCNonlinear
            buffer = NL
            psi = psi
            dealiasing = smooth_operator
            mobility = 1.0
        []

        # FFT for spectral solver
        [psi_hat]
            type = ForwardFFT
            buffer = psi_hat
            input = psi
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'psi'
    linear_reciprocal = 'linear'
    nonlinear_reciprocal = 'NL'
    reciprocal_buffer = 'psi_hat'
    corrector_order = 1
    corrector_steps = 3
    predictor_order = 1
    substeps = 10000
[]

[Postprocessors]
    [max_psi]
        type = TensorExtremeValuePostprocessor
        buffer = psi
        value_type = MAX
        execute_on = 'initial timestep_end'
    []
    [min_psi]
        type = TensorExtremeValuePostprocessor
        buffer = psi
        value_type = MIN
        execute_on = 'initial timestep_end'
    []
    [mean_psi]
        type = TensorAveragePostprocessor
        buffer = psi
        execute_on = 'initial timestep_end'
    []
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'psi'
        enable_hdf5 = true
    []
[]

[Executioner]
    type = Transient
    num_steps = 1000
    dt = 20  # Larger timestep for faster evolution
[]

[Outputs]
    perf_graph = true
    csv = true
[]
