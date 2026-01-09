[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '70e9 0.28' # Al
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/E'
  []
[]
