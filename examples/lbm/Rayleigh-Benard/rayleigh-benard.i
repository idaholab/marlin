Nx = 400
Ny = 200

TH = 1.1
TC = 1.0

frequency = '${Nx}/10.0'
amplitude = '${Ny}/100.0'

[Domain]
  dim = 2
  nx = '${Nx}'
  ny = '${Ny}'
  xmax = '${Nx}'
  ymax = '${Ny}'
  device_names='cpu'
  mesh_mode=DUMMY
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

!include buffers.i

[TensorComputes]

  #### Initialzie ####
  [Initialize]
    [density]
      type = LBMConstantTensor
      buffer = density
      constants = 'rho0'
    []

    [velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0 0.0'
    []

    [temperature]
      type = ParsedCompute
      buffer = T
      expression = 'a:=abs(y - sin(x / (${frequency} * pi)) * ${amplitude}) + y - sin(x / (${frequency} * pi)) * ${amplitude};
                    b:= a / (a + 1.0e-14);
                    ${TC} * b - b *${TH} + ${TH}'
      extra_symbols = true
    []

    [equilibrium_fluid]
      type = LBMEquilibrium
      buffer = feq
      bulk = density
      velocity = velocity
    []

    [equilibrium_fluid_total]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []

    [equilibrium_fluid_pc]
      type = LBMEquilibrium
      buffer = fpc
      bulk = density
      velocity = velocity
    []

    [equilibrium_temperature]
      type = LBMEquilibrium
      buffer = geq
      bulk = T
      velocity = velocity
    []

    [equilibrium_temperature_total]
      type = LBMEquilibrium
      buffer = g
      bulk = T
      velocity = velocity
    []

    [equilibrium_temperature_pc]
      type = LBMEquilibrium
      buffer = gpc
      bulk = T
      velocity = velocity
    []
  []

  !include solve.i
  !include boundary.i
[]

[TensorSolver]
  type = LBMStream
  buffer = 'f g'
  f_old = 'fpc gpc'
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'T velocity density'
    output_mode = 'Cell Cell Cell'
    enable_hdf5 = true
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  scalar_constant_names = 'rho0 T_0  T_C  T_H  tau_f tau_T  g'
  scalar_constant_values = '1.0  1.05  1.0  1.1  0.7   0.7   0.0001'
  substeps = 100
  print_debug_output = true
[]

[Executioner]
  type = Transient
  num_steps = 10000
[]
