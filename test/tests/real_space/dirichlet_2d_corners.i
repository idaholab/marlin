# 2D corner test: non-periodic on BOTH axes (Dirichlet on left/right, zero-flux
# Neumann on bottom/top), exercising the roadmap's documented ghost-fill
# axis-ordering caveat (TensorProblem::exchangeGhostLayers() fills one axis at
# a time, so a corner ghost cell filled while processing axis d may read
# transverse data whose own ghosts, from the other axis, have not been filled
# yet this call).
#
# The field u = 1 + 2*x has NO y-dependence, so it is compatible with uniform
# BC values on every face simultaneously: Dirichlet value = 1 + 2*x evaluated
# at x = 0 / x = 1 on the left/right faces (uniform there, since those faces
# are each at a single x), and zero-flux Neumann on the bottom/top faces is
# exactly satisfied since du/dy = 0 everywhere. (A genuinely bilinear profile,
# e.g. 1 + 2*x + y, is NOT uniform on any face -- it varies with x on the
# bottom/top faces and with y on the left/right faces -- so uniform Dirichlet
# values on all four faces can never exactly match it; an earlier version of
# this test used that profile and produced a large, correct-but-misleading
# Laplacian mismatch at the corners from the BC/IC inconsistency itself, not
# from ghost-fill ordering.) With this corrected, face-compatible profile, the
# discrete Laplacian should be exactly zero everywhere, including the four
# corner-adjacent cells, confirming the two-non-periodic-axis ghost-fill path
# (not exercised by any other real_space test, which are periodic on at least
# one axis) is correct at the corners.
[Domain]
  dim = 2
  nx = 20
  ny = 10
  parallel_mode = REAL_SPACE
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
[]

[Functions]
  [u_func]
    type = ParsedFunction
    expression = '1 + 2 * x'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      # Exactly the steady profile implied by the four BCs below (no
      # y-dependence), so the discrete Laplacian is checked instantaneously, no
      # time-stepping needed (cf. dirichlet_laplacian_check.i for the companion
      # periodic-in-y 1D case). MooseFunctionTensor evaluates this pointwise (no
      # symbolic simplification), so the profile can be written exactly, with no
      # perturbation term needed to work around ParsedCompute's spatial-axis
      # requirement.
      type = MooseFunctionTensor
      buffer = u
      function = u_func
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 3
    []
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux = 0
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux = 0
    []
  []
  [Solve]
    [lap_compute]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = lap
    []
  []
[]

[Postprocessors]
  [max_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MAX
  []
  [min_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MIN
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
