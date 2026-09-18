interface_width = 1.6
side = 80

gbe_max = '${units 0.65 J/m^2}'
L = '${fparse 1.0 * 1.6 / ${interface_width} }'

# ------------------------------------------------------------------
# Isotropic reference case.
#
# The GB energy is fixed at sigma = 0.65 J/m^2 for all boundaries,
# equal to gbe_max. Consequently gamma_ij = 1.5 for all pairs (the
# symmetric-well value), there are no torque terms, the mobility is
# uniform (L_NL == L), and the mobility-split correction
# (L_NL - L)*kappa*laplacian(gr_i) vanishes identically.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Periodic Voronoi-like IC for 5 grains.
#
# Each grain is seeded at a point (xi, yi). We use a smooth *periodic*
# distance metric based on sine functions,
#     d_i^2 = (side/pi)^2 * [ sin(pi*(x-xi)/side)^2 + sin(pi*(y-yi)/side)^2 ]
# which reduces to the Euclidean distance near the seed and wraps
# correctly across the periodic boundaries.
#
# The order parameters are then a softmax over the negative distances,
#     gr_i = exp(-a*d_i) / sum_j exp(-a*d_j),   a = 4/interface_width
# For any two competing grains this reduces exactly to the usual
# 0.5*(1 - tanh(2*(d_i - d_j)/interface_width)) profile, so the
# interfaces come out with the correct width, the partition of unity
# sum_i gr_i = 1 holds identically, and the cell structure is the
# (diffuse) Voronoi tessellation of the seeds under the periodic metric.
# ------------------------------------------------------------------
aexp = '${fparse 4.0 / ${interface_width} }'
lp = '${fparse ${side} / pi }'
kp = '${fparse pi / ${side} }'

# seed points (spread out over the periodic domain)
d0 = 'sqrt((${lp}*sin(${kp}*(x-8)))^2  + (${lp}*sin(${kp}*(y-6)))^2)'
d1 = 'sqrt((${lp}*sin(${kp}*(x-28)))^2 + (${lp}*sin(${kp}*(y-10)))^2)'
d2 = 'sqrt((${lp}*sin(${kp}*(x-20)))^2 + (${lp}*sin(${kp}*(y-24)))^2)'
d3 = 'sqrt((${lp}*sin(${kp}*(x-4)))^2  + (${lp}*sin(${kp}*(y-28)))^2)'
d4 = 'sqrt((${lp}*sin(${kp}*(x-34)))^2 + (${lp}*sin(${kp}*(y-33)))^2)'

g_gamma0 = '${fparse sqrt(2) / 3 }' # g(gamma=1.5)
f0_gamma0 = 0.1411
kappa = '${fparse gbe_max * interface_width * sqrt(f0_gamma0) / g_gamma0 }'
mu = '${fparse gbe_max  / (g_gamma0 * interface_width * sqrt(f0_gamma0) ) }'

gamma = 1.5

f0 = '((gr0^4/4 - gr0^2/2) + (gr1^4/4 - gr1^2/2) + (gr2^4/4 - gr2^2/2) + (gr3^4/4 - gr3^2/2) + (gr4^4/4 - gr4^2/2) + ${gamma}*(gr0^2*gr1^2 + gr0^2*gr2^2 + gr0^2*gr3^2 + gr0^2*gr4^2 + gr1^2*gr2^2 + gr1^2*gr3^2 + gr1^2*gr4^2 + gr2^2*gr3^2 + gr2^2*gr4^2 + gr3^2*gr4^2) + 0.25)'

[Domain]
    dim = 2
    nx = 200
    ny = 200
    xmax = ${side}
    ymax = ${side}
    mesh_mode = DUMMY
[]

[TensorComputes]
    [Initialize]
        # unnormalized softmax weights
        [c0]
            type = ParsedCompute
            buffer = c0
            expression = 'exp(-${aexp}*${d0})'
            extra_symbols = true
        []
        [c1]
            type = ParsedCompute
            buffer = c1
            expression = 'exp(-${aexp}*${d1})'
            extra_symbols = true
        []
        [c2]
            type = ParsedCompute
            buffer = c2
            expression = 'exp(-${aexp}*${d2})'
            extra_symbols = true
        []
        [c3]
            type = ParsedCompute
            buffer = c3
            expression = 'exp(-${aexp}*${d3})'
            extra_symbols = true
        []
        [c4]
            type = ParsedCompute
            buffer = c4
            expression = 'exp(-${aexp}*${d4})'
            extra_symbols = true
        []

        # normalized order parameters (partition of unity by construction)
        [gr0]
            type = ParsedCompute
            buffer = gr0
            expression = 'c0/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr1]
            type = ParsedCompute
            buffer = gr1
            expression = 'c1/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr2]
            type = ParsedCompute
            buffer = gr2
            expression = 'c2/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr3]
            type = ParsedCompute
            buffer = gr3
            expression = 'c3/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr4]
            type = ParsedCompute
            buffer = gr4
            expression = '1 - gr0 - gr1 - gr2 - gr3'
            inputs = 'gr0 gr1 gr2 gr3'
        []

        [L_kappa_laplacian]
            type = ReciprocalLaplacianFactor
            buffer = 'L_kappa_laplacian'
            factor = '${fparse ${L} * ${kappa}}'
        []
        [smooth]
            type = DeAliasingTensor
            method = HOULI
            buffer = smooth
        []
    []
    [Solve]
        [bnds]
            type = ParsedCompute
            buffer = 'bnds'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
            expression = 'gr0^2 + gr1^2 + gr2^2 + gr3^2 + gr4^2'
        []

        ## FFT
        [gr0_bar]
            type = ForwardFFT
            buffer = gr0_bar
            input = gr0
        []
        [gr1_bar]
            type = ForwardFFT
            buffer = gr1_bar
            input = gr1
        []
        [gr2_bar]
            type = ForwardFFT
            buffer = gr2_bar
            input = gr2
        []
        [gr3_bar]
            type = ForwardFFT
            buffer = gr3_bar
            input = gr3
        []
        [gr4_bar]
            type = ForwardFFT
            buffer = gr4_bar
            input = gr4
        []

        ## bulk driving forces (constant mobility L, constant gamma = ${gamma})
        [gr0_bulk_term]
            type = ParsedCompute
            buffer = 'gr0_bulk_term'
            expression = '-${L}*${mu} * (gr0^3 - gr0 + 2*${gamma}*gr0*(gr1^2 + gr2^2 + gr3^2 + gr4^2))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []
        [gr1_bulk_term]
            type = ParsedCompute
            buffer = 'gr1_bulk_term'
            expression = '-${L}*${mu} * (gr1^3 - gr1 + 2*${gamma}*gr1*(gr0^2 + gr2^2 + gr3^2 + gr4^2))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []
        [gr2_bulk_term]
            type = ParsedCompute
            buffer = 'gr2_bulk_term'
            expression = '-${L}*${mu} * (gr2^3 - gr2 + 2*${gamma}*gr2*(gr0^2 + gr1^2 + gr3^2 + gr4^2))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []
        [gr3_bulk_term]
            type = ParsedCompute
            buffer = 'gr3_bulk_term'
            expression = '-${L}*${mu} * (gr3^3 - gr3 + 2*${gamma}*gr3*(gr0^2 + gr1^2 + gr2^2 + gr4^2))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []
        [gr4_bulk_term]
            type = ParsedCompute
            buffer = 'gr4_bulk_term'
            expression = '-${L}*${mu} * (gr4^3 - gr4 + 2*${gamma}*gr4*(gr0^2 + gr1^2 + gr2^2 + gr3^2))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []

        [NL_gr0]
            type = ForwardFFT
            buffer = NL_gr0
            input = 'gr0_bulk_term'
        []
        [NL_gr1]
            type = ForwardFFT
            buffer = NL_gr1
            input = 'gr1_bulk_term'
        []
        [NL_gr2]
            type = ForwardFFT
            buffer = NL_gr2
            input = 'gr2_bulk_term'
        []
        [NL_gr3]
            type = ForwardFFT
            buffer = NL_gr3
            input = 'gr3_bulk_term'
        []
        [NL_gr4]
            type = ForwardFFT
            buffer = NL_gr4
            input = 'gr4_bulk_term'
        []

        [NL_gr0_smooth]
            type = ParsedCompute
            buffer = NL_gr0_smooth
            expression = 'smooth*NL_gr0'
            inputs = 'smooth NL_gr0'
        []
        [NL_gr1_smooth]
            type = ParsedCompute
            buffer = NL_gr1_smooth
            expression = 'smooth*NL_gr1'
            inputs = 'smooth NL_gr1'
        []
        [NL_gr2_smooth]
            type = ParsedCompute
            buffer = NL_gr2_smooth
            expression = 'smooth*NL_gr2'
            inputs = 'smooth NL_gr2'
        []
        [NL_gr3_smooth]
            type = ParsedCompute
            buffer = NL_gr3_smooth
            expression = 'smooth*NL_gr3'
            inputs = 'smooth NL_gr3'
        []
        [NL_gr4_smooth]
            type = ParsedCompute
            buffer = NL_gr4_smooth
            expression = 'smooth*NL_gr4'
            inputs = 'smooth NL_gr4'
        []

        # Properties for output
        [grain_ID]
            type = ParsedCompute
            buffer = 'grain_ID'
            expression = 'if(gr0>=max(gr1,max(gr2,max(gr3,gr4))), 0, if(gr1>=max(gr2,max(gr3,gr4)), 1, if(gr2>=max(gr3,gr4), 2, if(gr3>=gr4, 3, 4))))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
        []

        [grad_gr0]
            type = GradientVector
            buffer = grad_gr0
            input = gr0
        []
        [grad_gr1]
            type = GradientVector
            buffer = grad_gr1
            input = gr1
        []
        [grad_gr2]
            type = GradientVector
            buffer = grad_gr2
            input = gr2
        []
        [grad_gr3]
            type = GradientVector
            buffer = grad_gr3
            input = gr3
        []
        [grad_gr4]
            type = GradientVector
            buffer = grad_gr4
            input = gr4
        []

        [gradient_energy]
            type = GradientEnergyDensity
            buffer = gradient_energy
            gradient_buffers = 'grad_gr0 grad_gr1 grad_gr2 grad_gr3 grad_gr4'
            kappa = ${kappa}
        []
        [total_energy]
            type = ParsedCompute
            buffer = 'total_energy'
            expression = '${mu} * (${f0}) + gradient_energy'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gradient_energy'
        []
    []
[]

[Postprocessors]
    [total_gb_energy]
        type = TensorIntegralPostprocessor
        buffer = total_energy
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'gr0 gr1 gr2 gr3 gr4'
    reciprocal_buffer = 'gr0_bar gr1_bar gr2_bar gr3_bar gr4_bar'
    linear_reciprocal = 'L_kappa_laplacian L_kappa_laplacian L_kappa_laplacian L_kappa_laplacian L_kappa_laplacian'
    nonlinear_reciprocal = 'NL_gr0_smooth NL_gr1_smooth NL_gr2_smooth NL_gr3_smooth NL_gr4_smooth'
    substeps = 1000
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'gr0 gr1 gr2 gr3 gr4 bnds grain_ID'
        output_mode = 'NODE NODE NODE NODE NODE NODE CELL'
        enable_hdf5 = true
        transpose = true
    []
[]

[Executioner]
    type = Transient
    dt = 5
    num_steps = 50
[]

[Outputs]
    csv = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
