[Settings]
  require_double_precision = false
[]

[Models]
  [multiply]
    type = ScalarMultiplication
    from = 'A B'
    to = 'C'
  []
  [copy_vector]
    type = CopyVec
    from = input
    to = output
  []
  [symmetric]
    type = R2ToSR2
    input = input
    output = output
  []
[]

