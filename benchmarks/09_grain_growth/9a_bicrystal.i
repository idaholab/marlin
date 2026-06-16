#This simulates a bicrystal initial condition with a circular grain 1 (eta1)
#embedded in another grain (eta2). Curvature-driven interface motion causes
#eta1 to shrink away and disappear.
interface_width = 30
r0 = 400

[Domain]
  dim = 2
  nx = 200
  ny = 200
  xmin = -500
  xmax = 500
  ymin = -500
  ymax = 500
  mesh_mode = DUMMY
  device_names = 'mps'
[]

[TensorComputes]
    [Initialize]
        [eta1]
            type = ParsedCompute
            buffer = 'eta1'
            extra_symbols = 'true'
            expression = 'radius:=sqrt((x)^2+(y)^2);0.5 * (1 - tanh(2*(radius-${r0})/${interface_width}))'
        []
        [eta2]
            type = ParsedCompute
            buffer = 'eta2'
            extra_symbols = 'true'
            expression = 'radius:=sqrt((x)^2+(y)^2);0.5 * (1 + tanh(2*(radius-${r0})/${interface_width}))'
        []
        [kappa_linear_term]
            type = ReciprocalLaplacianFactor
            buffer = kappa_linear_term
            factor = 179.99863055 #L*kappa, L = 0.8545 nm^3/eV/microsec, kappa = 210.6479 eV/nm
        []
    []
    [Solve]
        [bulk_driving_force_1]
            type = ParsedCompute
            buffer = 'bulk_driving_force_1'
            constant_names = m
            constant_expressions = 1.8724 #units eV/nm^3
            expression = 'm * (eta1^4/4 - eta1^2/2 + eta2^4/4 - eta2^2/2 + 1.5*eta1^2*eta2^2 + 1/4)'
            inputs = 'eta1 eta2'
            derivatives = 'eta1'
        []
        [bulk_driving_force_2]
            type = ParsedCompute
            buffer = 'bulk_driving_force_2'
            constant_names = m
            constant_expressions = 1.8724 #units eV/nm^3
            expression = 'm * (eta1^4/4 - eta1^2/2 + eta2^4/4 - eta2^2/2 + 1.5*eta1^2*eta2^2 + 1/4)'
            inputs = 'eta1 eta2'
            derivatives = 'eta2'
        []

        [total_mu_driving_force_1]
            type = ParsedCompute
            buffer = 'total_mu_driving_force_1'
            inputs = 'bulk_driving_force_1'
            expression = '-0.8545 * bulk_driving_force_1' #L = 0.8545 nm^3/eV/microsec
        []
        [total_mu_driving_force_2]
            type = ParsedCompute
            buffer = 'total_mu_driving_force_2'
            inputs = 'bulk_driving_force_2'
            expression = '-0.8545 * bulk_driving_force_2' #L = 0.8545 nm^3/eV/microsec
        []

        [nonlinear1]
            type = ForwardFFT
            buffer = 'NL1'
            input = 'total_mu_driving_force_1'
        []
        [nonlinear2]
            type = ForwardFFT
            buffer = 'NL2'
            input = 'total_mu_driving_force_2'
        []

        [etabar1]
            type = ForwardFFT
            buffer = etabar1
            input = eta1
        []
        [etabar2]
            type = ForwardFFT
            buffer = etabar2
            input = eta2
        []
    []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'eta1 eta2'
  reciprocal_buffer = 'etabar1 etabar2'
  linear_reciprocal = 'kappa_linear_term kappa_linear_term'
  nonlinear_reciprocal = 'NL1 NL2'
  substeps = 1000
  corrector_steps = 1
  predictor_order = 1
  corrector_order = 1
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'eta1 eta2'
    enable_hdf5 = true
    transpose = false
  []
[]

[Postprocessors]
  [int_eta1]
    type = TensorIntegralPostprocessor
    buffer = eta1
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
    type = Transient
    dt = 5
    end_time = 500
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
