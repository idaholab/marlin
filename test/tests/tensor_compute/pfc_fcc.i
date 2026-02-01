# a0 = 3.52
pi = 3.14159265359
N = 50
psi_mean = -0.06

[Domain]
    dim = 2
    nx = ${N}
    ny = ${N}
    xmax = ${fparse ${N} * 2 * ${pi} * sqrt(3) / 16 }
    ymax = ${fparse ${N} * 2 * ${pi} * sqrt(3) / 16 }
    device_names = 'cpu'
[]

[TensorComputes]
    [Initialize]
        [psi]
            type = RandomTensor
            buffer = psi
            max = ${fparse ${psi_mean} + 1e-3 }
            min = ${fparse ${psi_mean} - 1e-3 }            
        []
        [linear]
            type = FCCPFCLinear
            buffer = 'linear'
            eps = 0.0082
            R1 = 0.0
            Q1 = ${fparse sqrt(4 / 3 ) }
        []
        [smooth_operator]
            type = DeAliasingTensor
            buffer = smooth_operator
            method = HOULI
        []
    []
    [Solve]
      [nl_div_psi_cubed]
        type = FCCPFCNonlinear
        buffer = NL
        psi = psi
        # dealiasing = smooth_operator
      []
    #   [zero]
    #     type = ConstantReciprocalTensor
    #     buffer = NL
    #     real = 0.0
    #     # dealiasing = smooth_operator
    #   []
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
  substeps = 1000
[]

[Postprocessors]
  [max_psi]
    type = TensorExtremeValuePostprocessor
    buffer = psi
    value_type = MAX
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
    num_steps = 100 #1000
    dt = 1e-2
[]

[Outputs]
    perf_graph = true
[]
