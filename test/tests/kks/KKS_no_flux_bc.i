#
# Kim-Kim-Suzuki with no-flux BC imposed using the smooth boundary method (SBM), solved on a 2D grid.
# Mask tensor 'psi' supplies the mask for the solve region to the system.
# Note: c is not directly conserved here - the masked value (psi > 0.0)*c will however be conserved.
#


# Constants for Initial Conditions
r = 30
l = 4.2

# Initial condition function for order parameter
eta_IC = '0.5*(1-tanh(2*(sqrt(x^2+y^2)-${r})/${l}))'

# Phase-field model parameters
kappa_eta = 5
rho_sq = 2
w = 1
M = 5
L = 5
c0_a = 0.3
c0_b = 0.7

# Expressions for switching function and bulk Gibbs energy
h_eta = 'eta^3*(6*eta^2-15*eta+10)'
F = '${h_eta}*(${rho_sq}*((c - (1-${h_eta})*(${c0_b} - ${c0_a}))-${c0_a})^2) + (1-${h_eta})*(${rho_sq}*((c + (${h_eta})*(${c0_b} - ${c0_a}))-${c0_b})^2 ) + ${w}*(eta^2)*(1-eta)^2'


[Domain]
    dim = 2
    nx = 20
    ny = 20

    xmin = -50
    xmax = 50
    ymin = -50
    ymax = 50

    # run on a CUDA device (adjust this to `cpu` if not available)
    device_names = 'cpu'

    # automatically create a matching mesh
    mesh_mode = DUMMY
[]
[Functions]

    [psi_func]
        type = ParsedFunction
        expression = 'if(x<x_min-${l},0,if(x>x_min+${l},1,0.5-0.5*cos(pi*(x-(x_min-${l}))/2/${l}) )) * if(x<x_max-${l},1,if(x>x_max+${l},0,0.5+0.5*cos(pi*(x-(x_max-${l}))/2/${l}) ))'
        symbol_names = 'x_min x_max y_min y_max'
        symbol_values = '30 70 0 100'
    []
[]

[TensorComputes]
    [Initialize]
        [c_IC]
            type = ParsedCompute
            buffer = c
            expression = '0.6 + (${c0_a}-0.6)*${eta_IC}'
            extra_symbols = 'true'
        []
        [eta_IC]
            type = ParsedCompute
            buffer = eta
            expression = '${eta_IC}'
            extra_symbols = 'true'
        []
        [psi_init]
             type = MooseFunctionTensor
            function = psi_func
            buffer = psi
        []
        [zero]
            type = ConstantReciprocalTensor
            buffer = zero
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
            type = ConstantTensor
            buffer = L_kappa
            real = ${fparse  ${L} * ${kappa_eta} }
        []
    []
    [Solve]
        [cbar]
            type = ForwardFFT
            buffer = cbar
            input = c
        []
        [etabar]
            type = ForwardFFT
            buffer = etabar
            input = eta
        []
        [mu]
            type = ParsedCompute
            buffer = 'mu'
            expression = '${F}'
            inputs = 'c eta'
            derivatives = 'c'
        []
        [div_J]
            type = ReciprocalMatDiffusion
            buffer = 'div_J'
            chemical_potential = mu
            mobility = M
            psi = psi
        []
        [domega_chem_deta]
            type = ParsedCompute
            buffer = 'domega_chem_deta'
            expression = '${F} - mu*c'
            inputs = 'mu c eta'
            derivatives = 'eta'
        []
        [AC_bulk]
            type = ReciprocalAllenCahn
            buffer = AC_bulk
            dF_chem_deta = domega_chem_deta
            L = L
            psi = psi
        []
        [kappa_grad_eta]
            type = ReciprocalMatDiffusion
            buffer = 'kappa_grad_eta'
            chemical_potential = 'eta'
            mobility = 'L_kappa'
            psi = psi
        []
        [AC_bar]
            type = ParsedCompute
            buffer = AC_bar
            expression = 'kappa_grad_eta + AC_bulk'
            inputs = 'AC_bulk kappa_grad_eta'

        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'c eta'
    reciprocal_buffer = 'cbar etabar'
    linear_reciprocal = 'zero zero'
    nonlinear_reciprocal = 'div_J AC_bar'
    substeps = 1e3
    predictor_order = 3
[]

[Postprocessors]
    [total_C]
        type = TensorIntegralPostprocessor
        buffer = c
        execute_on = 'INITIAL TIMESTEP_END'
    []
    [total_eta]
        type = TensorIntegralPostprocessor
        buffer = eta
        execute_on = 'INITIAL TIMESTEP_END'
    []
[]

[Problem]
    type = TensorProblem
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta c mu psi'
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
