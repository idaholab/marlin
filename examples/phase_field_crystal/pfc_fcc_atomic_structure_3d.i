# 3D FCC PFC test showing atomic-scale density fluctuations
# Based on PhysRevE.81.061601 two-mode FCC PFC model

pi = 3.14159265359

# Grid parameters - 3D resolution
N = 64  # Grid points per dimension (64^3 = 262k points)

# PFC model parameters - LARGER epsilon for clearer atomic structure
psi_mean = -0.2         # Mean density
epsilon = 0.15          # Larger undercooling for stronger modulations
R1_param = 0.0          # Two-mode coupling (0 for max FCC stability)
Q1_param = ${fparse sqrt(4.0/3.0)}  # FCC wave number ratio

# Domain size: want several unit cells
# The (111) wavelength is 2*pi/q0 = 2*pi*sqrt(3/4) ≈ 5.44 (dimensionless)
# Fit about 6-8 wavelengths in each direction
Lx = ${fparse 8 * 2 * ${pi} / ${Q1_param}}
Ly = ${Lx}
Lz = ${Lx}

[Domain]
    dim = 3
    nx = ${N}
    ny = ${N}
    nz = ${N}
    xmax = ${Lx}
    ymax = ${Ly}
    zmax = ${Lz}
    device_names = 'mps'
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
    num_steps = 200  # Reduced for 3D (much more expensive)
    dt = 5
[]

[Outputs]
    perf_graph = true
    csv = true
[]
