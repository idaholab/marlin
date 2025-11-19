#
# Kim-Kim-Suzuki with Gibbs energy supplied by a torch model, solved on a 2D grid.
#


# Constants for Initial Conditions
r = 30
l = 4.2

# Initial condition function for order parameter
eta_IC = '0.5*(1-tanh(2*(sqrt(x^2+y^2)-${r})/${l}))'

# Phase-field model parameters
kappa_eta = 5
w = 1
M = 5
L = 5

# Expressions for switching function and bulk Gibbs energy
h_eta = 'eta^3*(6*eta^2-15*eta+10)'


[Domain]
    dim = 2
    nx = 50
    ny = 50

    xmin = -50
    xmax = 50
    ymin = -50
    ymax = 50

    # automatically create a matching mesh
    mesh_mode = DUMMY
[]

[TensorComputes]
    [Initialize]
        [c_IC]
            type = ParsedCompute
            buffer = c
            expression = '0.7 + (0.3-0.6)*${eta_IC}'
            extra_symbols = 'true'
            enable_jit = false
        []
        [eta_IC]
            type = ParsedCompute
            buffer = eta
            expression = '${eta_IC}'
            extra_symbols = 'true'
            enable_jit = false
        []
        [psi_init]
            type = ConstantTensor
            buffer = psi
            real = 1
        []

        [M]
            type = ConstantTensor
            buffer = M
            real = ${M}
        []
        [L]
            type = ConstantTensor
            buffer = L
            real = ${L}
        []
        [L_kappa]
            type = ReciprocalLaplacianFactor
            buffer = L_kappa
            factor = ${fparse  ${L} * ${kappa_eta} }
        []
        [h_eta_IC]
            type = ParsedCompute
            buffer = h_eta
            expression = '${h_eta}'
            inputs = eta
        []
        [G_func_IC]
            type = LibtorchGibbsEnergy
            buffer = 'G'
            phase_fractions = 'h_eta'
            concentrations = 'c'
            domega_detas = 'dG_dh'
            chem_pots = 'mu'
            libtorch_model_file = 'marlin:libtorch_gibbs_energy/torch_NN_gibbs_model.pt'
        []
    []
    [Solve]
        [h_eta]
            type = ParsedCompute
            buffer = h_eta
            expression = '${h_eta}'
            inputs = eta
        []
        [G_func]
            type = LibtorchGibbsEnergy
            buffer = 'G'
            phase_fractions = 'h_eta'
            concentrations = 'c'
            domega_detas = 'dG_dh'
            chem_pots = 'mu'
            libtorch_model_file = 'marlin:libtorch_gibbs_energy/torch_NN_gibbs_model.pt'
        []
        [dG_deta]
            type = ParsedCompute
            buffer = 'dG_deta'
            inputs = 'eta dG_dh'
            expression = 'dG_dh * ${h_eta} + ${w} * eta^2 * (1-eta^2)^2'
            derivatives = 'eta'
        []

        [etabar]
            type = ForwardFFT
            buffer = etabar
            input = eta
        []
        [AC_bulk]
            type = ReciprocalAllenCahn
            L = L
            buffer = AC_bulk
            dF_chem_deta = dG_deta
            psi = psi
        []
        [NL_eta]
            type = ParsedCompute
            buffer = NL_eta
            expression = 'AC_bulk '
            inputs = 'AC_bulk'
        []
        [cbar]
            type = ForwardFFT
            buffer = cbar
            input = c
        []
        [div_J]
            type = ReciprocalMatDiffusion
            buffer = 'div_J'
            chemical_potential = mu
            mobility = M
            psi = psi
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'c eta'
    reciprocal_buffer = 'cbar etabar'
    linear_reciprocal = '0 L_kappa'
    nonlinear_reciprocal = 'div_J NL_eta'
    substeps = 1e3
    predictor_order = 3
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta c mu psi dG_deta dG_dh G'
        enable_hdf5 = true
        transpose = false
    []
[]

[Executioner]
    type = Transient
    dt = 0.1
    num_steps = 10
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
