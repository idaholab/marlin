[Settings]
  require_double_precision = false
[]

[Models]
  [multiply]
    type = ScalarMultiplication
    from_var = 'forces/A forces/B'
    to_var = 'state/C'
  []

  [rate]
    type = ScalarVariableRate
    time = forces/t
    variable = forces/A
    rate = state/dAdt
  []
[]

