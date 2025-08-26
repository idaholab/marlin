l = 10
sigma = 4
kappa = '(3*sigma*${l}/4)'
kappa_s = '40'
mu = '(6*${sigma}/${l})'

eta_IC = '((0.5 - 0.5*tanh(2*(x-150)/${l}))+(0.5 + 0.5*tanh(2*(x-250)/${l})))'
h_eta = '(eta^2/(eta^2 + (1-eta)^2))'
[Domain]
    dim = 2
    nx = 160
    ny = 40
    xmax = 400
    ymax = 100
    # run on a CUDA device (adjust this to `cpu` if not available)
    device_names = 'cpu'
    mesh_mode = DUMMY
[]

[TensorComputes]
    [Initialize]
        [c_Fe]
            type = ParsedCompute
            expression = '0.7*${eta_IC} + 0*(1-${eta_IC})'
            buffer = c_Fe
            extra_symbols = true
            expand = REAL
        []
        [c_Ni]
            type = ParsedCompute
            expression = '0.13*${eta_IC} + 0*(1-${eta_IC})'
            buffer = c_Ni
            extra_symbols = true
            expand = REAL
        []
        [c_Cr]
            type = ParsedCompute
            expression = '0.1699999*${eta_IC} + 0*(1-${eta_IC})'
            buffer = c_Cr
            extra_symbols = true
            expand = REAL
        []
        [eta]
            type = ParsedCompute
            expression = '${eta_IC}'
            extra_symbols = true
            buffer = eta
            expand = REAL
        []
        [L]
            type = ConstantTensor
            buffer = 'L'
            real = 0.1
        []
        [L_kappa]
            type = ReciprocalLaplacianFactor
            buffer = L_kappa
            factor = '${fparse 0.1 * ${kappa_s}}'
        []
        [L_kappa_s]
            type = ReciprocalLaplacianFactor
            buffer = L_kappa_s
            factor = '${fparse 0.1 * ( ${kappa_s} - ${kappa} ) }'
        []

        [psi]
            type = ConstantTensor
            buffer = psi
            real = 1
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
            phase_fractions = 'eta'
            domega_detas = 'dG_dh'
            concentrations = 'c_Fe c_Ni c_Cr'
            chem_pots = 'mu_Fe mu_Ni mu_Cr'
            libtorch_model_file = '/Users/bhavcv/projects/neams_msc/python_scripts/CALPHAD_thermodynamics/ternary_NN_final.pt'
            # libtorch_model_file = '/Users/bhavcv/projects/neams_msc/python_scripts/CALPHAD_thermodynamics/two_phase_energy.pt'
        []
    []
    [Solve]
        [h_eta]
            type = ParsedCompute
            buffer = h_eta
            expression = '${h_eta}'
            inputs = eta
        []
        # [dh_eta_deta]
        #     type = ParsedCompute
        #     buffer = dh_eta_deta
        #     expression = '${h_eta}'
        #     inputs = eta
        #     derivatives = eta
        # []
        [G_func]
            type = LibtorchGibbsEnergy
            buffer = 'G'
            phase_fractions = 'h_eta'
            domega_detas = 'dG_dh'
            concentrations = 'c_Fe c_Ni c_Cr'
            chem_pots = 'mu_Fe mu_Ni mu_Cr'
            libtorch_model_file = '/Users/bhavcv/projects/neams_msc/python_scripts/CALPHAD_thermodynamics/ternary_NN_final.pt'
        []
        [dG_deta]
            type = ParsedCompute
            buffer = 'dG_deta'
            inputs = 'eta dG_dh'
            expression = 'dG_dh * ${h_eta} + ${mu}*eta^2*(1-eta^2)' #
            derivatives = 'eta'
            expand = REAL
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
            expression = 'AC_bulk - etabar*L_kappa_s'
            inputs = 'etabar L_kappa_s AC_bulk'
        []
    []
[]

# [TensorSolver]
#     type = AdamsBashforthMoulton
#     buffer = 'eta'
#     reciprocal_buffer = 'etabar'
#     linear_reciprocal = 'L_kappa'
#     nonlinear_reciprocal = 'NL_eta'
#     substeps = 1e3
#     corrector_order = 3
#     predictor_order = 3
# []

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta c_Fe c_Cr c_Ni  dG_deta G h_eta mu_Cr mu_Fe mu_Ni dG_dh'#  ' #
        enable_hdf5 = true
        transpose = false
    []
[]

[Problem]
    type = TensorProblem
[]
[Executioner]
    type = Transient
    dt = 0.01
    num_steps = 5
[]
[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]