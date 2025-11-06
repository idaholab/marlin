#
# Test differentiation of local variables in ParsedCompute
#
# We test the expression with a buffer input: r := sqrt(a^2 + 1); r^2
# The derivative with respect to a should be: d/da(r^2) = 2*r * dr/da = 2*r * a/r = 2*a
#

[Domain]
  dim = 2
  nx = 20
  ny = 20
  xmax = 2
  ymax = 2
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [a]        # Input variable
  []
  [df_da]    # Auto-differentiated derivative
  []
  [df_da_exact]  # Hand-coded exact derivative
  []
  [error]    # Absolute difference
  []
[]

[TensorComputes]
  [Solve]
    # Initialize input buffer
    [init_a]
      type = ParsedCompute
      buffer = a
      expression = 'x + 0.5*y'
      extra_symbols = true
    []

    # Auto-differentiated derivative: d/da(r^2) where r:=sqrt(a^2+1)
    [auto_derivative]
      type = ParsedCompute
      buffer = df_da
      expression = 'r:=sqrt(a^2+1); r^2'
      derivatives = 'a'
      inputs = 'a'
    []

    # Hand-coded exact derivative: d/da(r^2) = 2*a
    [exact_derivative]
      type = ParsedCompute
      buffer = df_da_exact
      expression = '2*a'
      inputs = 'a'
    []

    # Compute absolute error
    [compute_error]
      type = ParsedCompute
      buffer = error
      expression = 'abs(df_da - df_da_exact)'
      inputs = 'df_da df_da_exact'
    []
  []
[]

[Postprocessors]
  [integral_error]
    type = TensorIntegralPostprocessor
    buffer = error
    execute_on = 'INITIAL'
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 0
[]

[Outputs]
  csv = true
[]
