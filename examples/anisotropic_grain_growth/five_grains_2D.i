interface_width = 1.6
side = 80

gbe_max = '${units 1.24 J/m^2}'
L = '${fparse 1.0 * 1.6 / ${interface_width} }'

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

# seed points (spread out over the periodic 40x40 domain)
d0 = 'sqrt((${lp}*sin(${kp}*(x-8)))^2  + (${lp}*sin(${kp}*(y-6)))^2)'
d1 = 'sqrt((${lp}*sin(${kp}*(x-28)))^2 + (${lp}*sin(${kp}*(y-10)))^2)'
d2 = 'sqrt((${lp}*sin(${kp}*(x-20)))^2 + (${lp}*sin(${kp}*(y-24)))^2)'
d3 = 'sqrt((${lp}*sin(${kp}*(x-4)))^2  + (${lp}*sin(${kp}*(y-28)))^2)'
d4 = 'sqrt((${lp}*sin(${kp}*(x-34)))^2 + (${lp}*sin(${kp}*(y-33)))^2)'

g_gamma0 = '${fparse sqrt(2) / 3 }' # g(gamma=1.5)
f0_gamma0 = 0.1411
kappa = '${fparse gbe_max * interface_width * sqrt(f0_gamma0) / g_gamma0 }'
mu = '${fparse gbe_max  / (g_gamma0 * interface_width * sqrt(f0_gamma0) ) }'

g_to_gamma_func = '(1.0/(-3.0944 * g^8 -1.8169*g^6 + 10.323 * g^4 - 8.1819*g^2 + 2.0033))'
dgamma_dsigma = '(g * (24.7552*g^6 + 10.9014*g^4 - 41.292*g^2 + 16.3638) / (-3.0944*g^8 - 1.8169*g^6 + 10.323*g^4 - 8.1819*g^2 + 2.0033)^2 / sqrt(${kappa} * ${mu}) )'

f0 = '((gr0^4/4 - gr0^2/2) + (gr1^4/4 - gr1^2/2) + (gr2^4/4 - gr2^2/2) + (gr3^4/4 - gr3^2/2) + (gr4^4/4 - gr4^2/2) + gamma_01*gr0^2*gr1^2 + gamma_02*gr0^2*gr2^2 + gamma_03*gr0^2*gr3^2 + gamma_04*gr0^2*gr4^2 + gamma_12*gr1^2*gr2^2 + gamma_13*gr1^2*gr3^2 + gamma_14*gr1^2*gr4^2 + gamma_23*gr2^2*gr3^2 + gamma_24*gr2^2*gr4^2 + gamma_34*gr3^2*gr4^2 + 0.25)'

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
        [kappa_laplacian]
            type = ReciprocalLaplacianFactor
            buffer = kappa_laplacian
            factor = '${kappa}'
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

        ## Pairwise anisotropic GB energies (10 pairs)
        # NOTE: the per-pair libtorch misorientation models (Ri_Rj.pt) will be
        # developed separately; placeholder file names below follow the
        # existing convention and location.
        [sigma_gr0_gr1]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr1'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr1_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr1_dgrad_gr1'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr1'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R1_R2.pt'
            chunk_size = 1e5
        []
        [sigma_gr0_gr2]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr2'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr2_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr2_dgrad_gr2'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr2'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R1_R3.pt'
            chunk_size = 1e5
        []
        [sigma_gr0_gr3]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr3'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr3_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr3_dgrad_gr3'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr3'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R1_R4.pt'
            chunk_size = 1e5
        []
        [sigma_gr0_gr4]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr4'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr4_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr4_dgrad_gr4'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr4'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R1_R5.pt'
            chunk_size = 1e5
        []
        [sigma_gr1_gr2]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr1_gr2'
            dsigma_dgrad_grain1 = 'dsigma_gr1_gr2_dgrad_gr1'
            dsigma_dgrad_grain2 = 'dsigma_gr1_gr2_dgrad_gr2'
            grad_grain1_buffer = 'grad_gr1'
            grad_grain2_buffer = 'grad_gr2'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R2_R3.pt'
            chunk_size = 1e5
        []
        [sigma_gr1_gr3]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr1_gr3'
            dsigma_dgrad_grain1 = 'dsigma_gr1_gr3_dgrad_gr1'
            dsigma_dgrad_grain2 = 'dsigma_gr1_gr3_dgrad_gr3'
            grad_grain1_buffer = 'grad_gr1'
            grad_grain2_buffer = 'grad_gr3'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R2_R4.pt'
            chunk_size = 1e5
        []
        [sigma_gr1_gr4]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr1_gr4'
            dsigma_dgrad_grain1 = 'dsigma_gr1_gr4_dgrad_gr1'
            dsigma_dgrad_grain2 = 'dsigma_gr1_gr4_dgrad_gr4'
            grad_grain1_buffer = 'grad_gr1'
            grad_grain2_buffer = 'grad_gr4'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R2_R5.pt'
            chunk_size = 1e5
        []
        [sigma_gr2_gr3]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr2_gr3'
            dsigma_dgrad_grain1 = 'dsigma_gr2_gr3_dgrad_gr2'
            dsigma_dgrad_grain2 = 'dsigma_gr2_gr3_dgrad_gr3'
            grad_grain1_buffer = 'grad_gr2'
            grad_grain2_buffer = 'grad_gr3'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R3_R4.pt'
            chunk_size = 1e5
        []
        [sigma_gr2_gr4]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr2_gr4'
            dsigma_dgrad_grain1 = 'dsigma_gr2_gr4_dgrad_gr2'
            dsigma_dgrad_grain2 = 'dsigma_gr2_gr4_dgrad_gr4'
            grad_grain1_buffer = 'grad_gr2'
            grad_grain2_buffer = 'grad_gr4'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R3_R5.pt'
            chunk_size = 1e5
        []
        [sigma_gr3_gr4]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr3_gr4'
            dsigma_dgrad_grain1 = 'dsigma_gr3_gr4_dgrad_gr3'
            dsigma_dgrad_grain2 = 'dsigma_gr3_gr4_dgrad_gr4'
            grad_grain1_buffer = 'grad_gr3'
            grad_grain2_buffer = 'grad_gr4'
            interface_width = '${interface_width}'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_R4_R5.pt'
            chunk_size = 1e5
        []

        ## gamma_ij(sigma_ij)
        [gamma_01]
            type = ParsedCompute
            buffer = 'gamma_01'
            expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr1'
        []
        [gamma_02]
            type = ParsedCompute
            buffer = 'gamma_02'
            expression = 'g:=sigma_gr0_gr2/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr2'
        []
        [gamma_03]
            type = ParsedCompute
            buffer = 'gamma_03'
            expression = 'g:=sigma_gr0_gr3/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr3'
        []
        [gamma_04]
            type = ParsedCompute
            buffer = 'gamma_04'
            expression = 'g:=sigma_gr0_gr4/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr0_gr4'
        []
        [gamma_12]
            type = ParsedCompute
            buffer = 'gamma_12'
            expression = 'g:=sigma_gr1_gr2/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr1_gr2'
        []
        [gamma_13]
            type = ParsedCompute
            buffer = 'gamma_13'
            expression = 'g:=sigma_gr1_gr3/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr1_gr3'
        []
        [gamma_14]
            type = ParsedCompute
            buffer = 'gamma_14'
            expression = 'g:=sigma_gr1_gr4/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr1_gr4'
        []
        [gamma_23]
            type = ParsedCompute
            buffer = 'gamma_23'
            expression = 'g:=sigma_gr2_gr3/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr2_gr3'
        []
        [gamma_24]
            type = ParsedCompute
            buffer = 'gamma_24'
            expression = 'g:=sigma_gr2_gr4/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr2_gr4'
        []
        [gamma_34]
            type = ParsedCompute
            buffer = 'gamma_34'
            expression = 'g:=sigma_gr3_gr4/sqrt(${kappa}*${mu});${g_to_gamma_func}'
            inputs = 'sigma_gr3_gr4'
        []

        ## torque prefactors dgamma/dsigma * gr_i^2 * gr_j^2
        [coeff_01]
            type = ParsedCompute
            buffer = 'coeff_01'
            expression = 'g:=sigma_gr0_gr1/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr1^2'
            inputs = 'sigma_gr0_gr1 gr0 gr1'
        []
        [coeff_02]
            type = ParsedCompute
            buffer = 'coeff_02'
            expression = 'g:=sigma_gr0_gr2/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr2^2'
            inputs = 'sigma_gr0_gr2 gr0 gr2'
        []
        [coeff_03]
            type = ParsedCompute
            buffer = 'coeff_03'
            expression = 'g:=sigma_gr0_gr3/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr3^2'
            inputs = 'sigma_gr0_gr3 gr0 gr3'
        []
        [coeff_04]
            type = ParsedCompute
            buffer = 'coeff_04'
            expression = 'g:=sigma_gr0_gr4/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr0^2 *gr4^2'
            inputs = 'sigma_gr0_gr4 gr0 gr4'
        []
        [coeff_12]
            type = ParsedCompute
            buffer = 'coeff_12'
            expression = 'g:=sigma_gr1_gr2/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr1^2 *gr2^2'
            inputs = 'sigma_gr1_gr2 gr1 gr2'
        []
        [coeff_13]
            type = ParsedCompute
            buffer = 'coeff_13'
            expression = 'g:=sigma_gr1_gr3/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr1^2 *gr3^2'
            inputs = 'sigma_gr1_gr3 gr1 gr3'
        []
        [coeff_14]
            type = ParsedCompute
            buffer = 'coeff_14'
            expression = 'g:=sigma_gr1_gr4/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr1^2 *gr4^2'
            inputs = 'sigma_gr1_gr4 gr1 gr4'
        []
        [coeff_23]
            type = ParsedCompute
            buffer = 'coeff_23'
            expression = 'g:=sigma_gr2_gr3/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr2^2 *gr3^2'
            inputs = 'sigma_gr2_gr3 gr2 gr3'
        []
        [coeff_24]
            type = ParsedCompute
            buffer = 'coeff_24'
            expression = 'g:=sigma_gr2_gr4/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr2^2 *gr4^2'
            inputs = 'sigma_gr2_gr4 gr2 gr4'
        []
        [coeff_34]
            type = ParsedCompute
            buffer = 'coeff_34'
            expression = 'g:=sigma_gr3_gr4/sqrt(${kappa}*${mu});${dgamma_dsigma} * gr3^2 *gr4^2'
            inputs = 'sigma_gr3_gr4 gr3 gr4'
        []

        ## anisotropy torques (2 per pair)
        [torque_01_gr0]
            type = AnisotropyTorque
            buffer = torque_01_gr0
            dmu_dn = dsigma_gr0_gr1_dgrad_gr0
            g = 'coeff_01'
        []
        [torque_01_gr1]
            type = AnisotropyTorque
            buffer = torque_01_gr1
            dmu_dn = dsigma_gr0_gr1_dgrad_gr1
            g = 'coeff_01'
        []
        [torque_02_gr0]
            type = AnisotropyTorque
            buffer = torque_02_gr0
            dmu_dn = dsigma_gr0_gr2_dgrad_gr0
            g = 'coeff_02'
        []
        [torque_02_gr2]
            type = AnisotropyTorque
            buffer = torque_02_gr2
            dmu_dn = dsigma_gr0_gr2_dgrad_gr2
            g = 'coeff_02'
        []
        [torque_03_gr0]
            type = AnisotropyTorque
            buffer = torque_03_gr0
            dmu_dn = dsigma_gr0_gr3_dgrad_gr0
            g = 'coeff_03'
        []
        [torque_03_gr3]
            type = AnisotropyTorque
            buffer = torque_03_gr3
            dmu_dn = dsigma_gr0_gr3_dgrad_gr3
            g = 'coeff_03'
        []
        [torque_04_gr0]
            type = AnisotropyTorque
            buffer = torque_04_gr0
            dmu_dn = dsigma_gr0_gr4_dgrad_gr0
            g = 'coeff_04'
        []
        [torque_04_gr4]
            type = AnisotropyTorque
            buffer = torque_04_gr4
            dmu_dn = dsigma_gr0_gr4_dgrad_gr4
            g = 'coeff_04'
        []
        [torque_12_gr1]
            type = AnisotropyTorque
            buffer = torque_12_gr1
            dmu_dn = dsigma_gr1_gr2_dgrad_gr1
            g = 'coeff_12'
        []
        [torque_12_gr2]
            type = AnisotropyTorque
            buffer = torque_12_gr2
            dmu_dn = dsigma_gr1_gr2_dgrad_gr2
            g = 'coeff_12'
        []
        [torque_13_gr1]
            type = AnisotropyTorque
            buffer = torque_13_gr1
            dmu_dn = dsigma_gr1_gr3_dgrad_gr1
            g = 'coeff_13'
        []
        [torque_13_gr3]
            type = AnisotropyTorque
            buffer = torque_13_gr3
            dmu_dn = dsigma_gr1_gr3_dgrad_gr3
            g = 'coeff_13'
        []
        [torque_14_gr1]
            type = AnisotropyTorque
            buffer = torque_14_gr1
            dmu_dn = dsigma_gr1_gr4_dgrad_gr1
            g = 'coeff_14'
        []
        [torque_14_gr4]
            type = AnisotropyTorque
            buffer = torque_14_gr4
            dmu_dn = dsigma_gr1_gr4_dgrad_gr4
            g = 'coeff_14'
        []
        [torque_23_gr2]
            type = AnisotropyTorque
            buffer = torque_23_gr2
            dmu_dn = dsigma_gr2_gr3_dgrad_gr2
            g = 'coeff_23'
        []
        [torque_23_gr3]
            type = AnisotropyTorque
            buffer = torque_23_gr3
            dmu_dn = dsigma_gr2_gr3_dgrad_gr3
            g = 'coeff_23'
        []
        [torque_24_gr2]
            type = AnisotropyTorque
            buffer = torque_24_gr2
            dmu_dn = dsigma_gr2_gr4_dgrad_gr2
            g = 'coeff_24'
        []
        [torque_24_gr4]
            type = AnisotropyTorque
            buffer = torque_24_gr4
            dmu_dn = dsigma_gr2_gr4_dgrad_gr4
            g = 'coeff_24'
        []
        [torque_34_gr3]
            type = AnisotropyTorque
            buffer = torque_34_gr3
            dmu_dn = dsigma_gr3_gr4_dgrad_gr3
            g = 'coeff_34'
        []
        [torque_34_gr4]
            type = AnisotropyTorque
            buffer = torque_34_gr4
            dmu_dn = dsigma_gr3_gr4_dgrad_gr4
            g = 'coeff_34'
        []

        ## interpolated local GB energy and nonlinear mobility
        [sigma]
            type = ParsedCompute
            buffer = 'sigma'
            expression = '(gr0^2*gr1^2*sigma_gr0_gr1 + gr0^2*gr2^2*sigma_gr0_gr2 + gr0^2*gr3^2*sigma_gr0_gr3 + gr0^2*gr4^2*sigma_gr0_gr4 + gr1^2*gr2^2*sigma_gr1_gr2 + gr1^2*gr3^2*sigma_gr1_gr3 + gr1^2*gr4^2*sigma_gr1_gr4 + gr2^2*gr3^2*sigma_gr2_gr3 + gr2^2*gr4^2*sigma_gr2_gr4 + gr3^2*gr4^2*sigma_gr3_gr4)/(gr0^2*gr1^2 + gr0^2*gr2^2 + gr0^2*gr3^2 + gr0^2*gr4^2 + gr1^2*gr2^2 + gr1^2*gr3^2 + gr1^2*gr4^2 + gr2^2*gr3^2 + gr2^2*gr4^2 + gr3^2*gr4^2 + 1e-4)'
            inputs = 'sigma_gr0_gr1 sigma_gr0_gr2 sigma_gr0_gr3 sigma_gr0_gr4 sigma_gr1_gr2 sigma_gr1_gr3 sigma_gr1_gr4 sigma_gr2_gr3 sigma_gr2_gr4 sigma_gr3_gr4 gr0 gr1 gr2 gr3 gr4'
        []
        [L_NL]
            type = ParsedCompute
            buffer = 'L_NL'
            expression = '${L} * sigma / ${gbe_max}'
            inputs = 'sigma'
        []

        ## kappa * laplacian(gr_i) in real space (for the mobility-split correction)
        [kappa_laplacian_gr0_hat]
            type = ParsedCompute
            buffer = kappa_laplacian_gr0_hat
            inputs = 'gr0_bar kappa_laplacian'
            expression = 'gr0_bar * kappa_laplacian'
        []
        [kappa_laplacian_gr1_hat]
            type = ParsedCompute
            buffer = kappa_laplacian_gr1_hat
            inputs = 'gr1_bar kappa_laplacian'
            expression = 'gr1_bar * kappa_laplacian'
        []
        [kappa_laplacian_gr2_hat]
            type = ParsedCompute
            buffer = kappa_laplacian_gr2_hat
            inputs = 'gr2_bar kappa_laplacian'
            expression = 'gr2_bar * kappa_laplacian'
        []
        [kappa_laplacian_gr3_hat]
            type = ParsedCompute
            buffer = kappa_laplacian_gr3_hat
            inputs = 'gr3_bar kappa_laplacian'
            expression = 'gr3_bar * kappa_laplacian'
        []
        [kappa_laplacian_gr4_hat]
            type = ParsedCompute
            buffer = kappa_laplacian_gr4_hat
            inputs = 'gr4_bar kappa_laplacian'
            expression = 'gr4_bar * kappa_laplacian'
        []

        [kappa_laplacian_gr0]
            type = InverseFFT
            buffer = kappa_laplacian_gr0
            input = kappa_laplacian_gr0_hat
        []
        [kappa_laplacian_gr1]
            type = InverseFFT
            buffer = kappa_laplacian_gr1
            input = kappa_laplacian_gr1_hat
        []
        [kappa_laplacian_gr2]
            type = InverseFFT
            buffer = kappa_laplacian_gr2
            input = kappa_laplacian_gr2_hat
        []
        [kappa_laplacian_gr3]
            type = InverseFFT
            buffer = kappa_laplacian_gr3
            input = kappa_laplacian_gr3_hat
        []
        [kappa_laplacian_gr4]
            type = InverseFFT
            buffer = kappa_laplacian_gr4
            input = kappa_laplacian_gr4_hat
        []

        ## bulk driving forces
        [gr0_bulk_term]
            type = ParsedCompute
            buffer = 'gr0_bulk_term'
            expression = '-L_NL*${mu} * (gr0^3 - gr0 + 2*gr0*(gamma_01*gr1^2 + gamma_02*gr2^2 + gamma_03*gr3^2 + gamma_04*gr4^2) - torque_01_gr0 - torque_02_gr0 - torque_03_gr0 - torque_04_gr0) + (L_NL - ${L}) * kappa_laplacian_gr0'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_01 gamma_02 gamma_03 gamma_04 torque_01_gr0 torque_02_gr0 torque_03_gr0 torque_04_gr0 L_NL kappa_laplacian_gr0'
        []
        [gr1_bulk_term]
            type = ParsedCompute
            buffer = 'gr1_bulk_term'
            expression = '-L_NL*${mu} * (gr1^3 - gr1 + 2*gr1*(gamma_01*gr0^2 + gamma_12*gr2^2 + gamma_13*gr3^2 + gamma_14*gr4^2) - torque_01_gr1 - torque_12_gr1 - torque_13_gr1 - torque_14_gr1) + (L_NL - ${L}) * kappa_laplacian_gr1'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_01 gamma_12 gamma_13 gamma_14 torque_01_gr1 torque_12_gr1 torque_13_gr1 torque_14_gr1 L_NL kappa_laplacian_gr1'
        []
        [gr2_bulk_term]
            type = ParsedCompute
            buffer = 'gr2_bulk_term'
            expression = '-L_NL*${mu} * (gr2^3 - gr2 + 2*gr2*(gamma_02*gr0^2 + gamma_12*gr1^2 + gamma_23*gr3^2 + gamma_24*gr4^2) - torque_02_gr2 - torque_12_gr2 - torque_23_gr2 - torque_24_gr2) + (L_NL - ${L}) * kappa_laplacian_gr2'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_02 gamma_12 gamma_23 gamma_24 torque_02_gr2 torque_12_gr2 torque_23_gr2 torque_24_gr2 L_NL kappa_laplacian_gr2'
        []
        [gr3_bulk_term]
            type = ParsedCompute
            buffer = 'gr3_bulk_term'
            expression = '-L_NL*${mu} * (gr3^3 - gr3 + 2*gr3*(gamma_03*gr0^2 + gamma_13*gr1^2 + gamma_23*gr2^2 + gamma_34*gr4^2) - torque_03_gr3 - torque_13_gr3 - torque_23_gr3 - torque_34_gr3) + (L_NL - ${L}) * kappa_laplacian_gr3'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_03 gamma_13 gamma_23 gamma_34 torque_03_gr3 torque_13_gr3 torque_23_gr3 torque_34_gr3 L_NL kappa_laplacian_gr3'
        []
        [gr4_bulk_term]
            type = ParsedCompute
            buffer = 'gr4_bulk_term'
            expression = '-L_NL*${mu} * (gr4^3 - gr4 + 2*gr4*(gamma_04*gr0^2 + gamma_14*gr1^2 + gamma_24*gr2^2 + gamma_34*gr3^2) - torque_04_gr4 - torque_14_gr4 - torque_24_gr4 - torque_34_gr4) + (L_NL - ${L}) * kappa_laplacian_gr4'
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_04 gamma_14 gamma_24 gamma_34 torque_04_gr4 torque_14_gr4 torque_24_gr4 torque_34_gr4 L_NL kappa_laplacian_gr4'
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
            inputs = 'gr0 gr1 gr2 gr3 gr4 gamma_01 gamma_02 gamma_03 gamma_04 gamma_12 gamma_13 gamma_14 gamma_23 gamma_24 gamma_34 gradient_energy'
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
        buffer = 'gr0 gr1 gr2 gr3 gr4 bnds sigma grain_ID'
        output_mode = 'NODE NODE NODE NODE NODE NODE NODE CELL'
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
