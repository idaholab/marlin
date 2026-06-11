import numpy as np
from scipy import ndimage
from skimage import morphology
import porespy as ps

# Collection of functions for processing pore structures

def thermodynamic_properties(Pi_phy, Ti_phy, Tc, Pc, w, Mw, d_mole):
    """
    Calculate thermodynamic properties for gasses using Peng-Robinson equation of state.

    Parameters:
        Pi_phy: Pressure in Pa
        Ti_phy: Temperature in K
        Tc: Critical temperature in K
        Pc: Critical pressure in Pa
        w: Acentric factor (dimensionless)
        Mw: Molecular weight in kg/mol
        d_mole: Molecular diameter in m

    Returns:
        rhoG_phy: Gas density in kg/m³
        rhoG_lu: Gas density in lattice units
        lamb: Mean free path in m
        Crho: Density conversion factor from physical to lattice units
    """

    # Critical temperature, pressure and molar volume, and acentric factors
    # PR EOS parameters for components
    R = 8.31446  # M3*Pa/K/mol
    k_component = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alpha_component = (1 + k_component * (1 - np.sqrt(Ti_phy / Tc)))**2
    a_component = 0.45724 * (R * Tc)**2 / Pc
    a_alpha_component = a_component * alpha_component
    b_component = 0.07780 * R * Tc / Pc
    aG = a_alpha_component
    bG = b_component

    # Cubic equation of compressibility (Z) ← Pv=RT (n=1) & PR EOS
    AG = aG * Pi_phy / R**2 / Ti_phy**2
    BG = bG * Pi_phy / R / Ti_phy

    # Z3*Z^3 + Z2*Z^2 + Z1*Z + Z0 = 0
    ZG3 = 1
    ZG2 = BG - 1
    ZG1 = AG - 3 * BG**2 - 2 * BG
    ZG0 = BG**3 + BG**2 - AG * BG
    ZG_results = np.roots([ZG3, ZG2, ZG1, ZG0])
    ZG_real = ZG_results[np.isreal(ZG_results)].real
    ZG = np.max(ZG_real)
    vG = ZG * R * Ti_phy / Pi_phy

    rhoG_phy_mol = 1 / vG  # mol/m^3 - molar density

    Rs = R / Mw
    rhoG_phy = rhoG_phy_mol * Mw  # kg/m^3

    b_Rs_component = 0.07780 * Rs * Tc / Pc  # m3/kg
    bs_lu = 2 / 21
    Mw_lu = 1.0
    C_bs = b_Rs_component / bs_lu  # (m3/kg)/(m_lu3/kg_lu)
    Crho = 1 / C_bs
    C_bs = b_Rs_component / bs_lu
    C_Mw = Mw / Mw_lu

    Crho_mol = 1 / (C_bs * C_Mw)  # conversion factor of mole density, mol/m3
    rhoG_lu = rhoG_phy / Crho  # kg/m^3

    # Mean free path calculation
    r_mole = d_mole / 2
    rho_mol = rhoG_lu / Mw_lu
    n = rho_mol * Crho_mol * 6.02214076e23
    lamb = 1 / (np.sqrt(2) * n * np.pi * (2 * r_mole + 2 * r_mole)**2)

    return rhoG_phy, rhoG_lu, lamb, Crho

def clean_boundaries(pore):
    """
    Clean boundary pixels that don't match valid boundary types.

    Iteratively removes solid pixels whose 3x3 neighborhood pattern
    doesn't correspond to a valid boundary type defined in bnd_types.
    Uses periodic boundary conditions.

    Parameters:
        pore: 2D numpy array (boolean or binary) representing the pore structure
              (1 = solid, 0 = void)

    Returns:
        pore: Cleaned 2D numpy array with invalid boundary pixels removed
    """
    #
    bnd_types = {
        31, 55, 63, 91, 95, 119, 127, 217, 219, 221, 223, 247, 253, 255,
        287, 310, 311, 319, 347, 351, 374, 375, 379, 382, 383, 415, 436,
        437, 438, 439, 445, 447, 472, 473, 475, 476, 477, 478, 479, 496,
        497, 499, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510, 511,
        1023, 2047, 4095, 8191, 16383, 32767, 65535
    }

    # Add periodic ghost layers
    ny, nx = pore.shape
    pore_perio = np.zeros((ny + 2, nx + 2), dtype=pore.dtype)
    pore_perio[1:-1, 1:-1] = pore

    # Columns (Periodic)
    pore_perio[1:-1, 0] = pore[:, -1]
    pore_perio[1:-1, -1] = pore[:, 0]

    # Rows (Periodic)
    pore_perio[0, 1:-1] = pore[-1, :]
    pore_perio[-1, 1:-1] = pore[0, :]

    # Corners (Explicit handling for 3x3 kernels)
    pore_perio[0, 0] = pore[-1, -1]
    pore_perio[0, -1] = pore[-1, 0]
    pore_perio[-1, 0] = pore[0, -1]
    pore_perio[-1, -1] = pore[0, 0]

    pore_perio_cp = pore_perio.copy()
    diff = 1
    cleaning_attempt = 1

    # Loop through the domain
    while diff > 0:
        print(f'Cleaning attempt- {cleaning_attempt}')
        cleaning_attempt += 1

        # Iterating strictly to match MATLAB's sequential update logic
        for i in range(1, ny + 1):
            for j in range(1, nx + 1):
                kernel = pore_perio[i-1:i+2, j-1:j+2]

                # Check center pixel and sum
                if kernel[1, 1] == 1 and np.sum(kernel) < 9:
                    # MATLAB reshape is column-major (Fortran order).
                    kernel_lin = kernel.flatten(order='F')

                    # binary vector to decimal
                    bin_str = "".join(kernel_lin.astype(int).astype(str))
                    decimal_number = int(bin_str, 2)

                    if decimal_number not in bnd_types:
                        pore_perio[i, j] = 0

        diff = np.sum(pore_perio_cp ^ pore_perio)  # XOR for boolean arrays
        pore_perio_cp = pore_perio.copy()

    return pore_perio[1:-1, 1:-1]

def pore_properties_opt(pore, lamb, Cl):
    """
    Calculate pore properties including Knudsen number and local pore size.
    Computes the medial axis (skeleton) of the pore space and calculates local pore sizes based on distance to walls.
    Iteratively removes pores that are too small.

    Parameters:
        pore: 2D numpy array (boolean or binary) representing the pore structure
              (1 = pore/void, 0 = solid)
        lamb: Mean free path in physical units (m)
        Cl: Length conversion factor from physical to lattice units

    Returns:
        Kn: 2D array of Knudsen numbers at each pore location
        locporesize: 2D array of local pore sizes (in lattice units)
        pore: Cleaned pore structure with small pores removed
        BW3: Medial axis with distance values (local pore size at midaxis)
    """
    pore = pore.copy()
    ny, nx = pore.shape
    diff = 1.0
    cleaning_attempt = 1

    while diff > 0:
        print(f'Smoothing pores - {cleaning_attempt}')
        cleaning_attempt += 1
        pore_cp = pore.copy()

        w_bnd = max(ny, nx)
        pore_expand = np.ones((ny + 2 * w_bnd, nx + 2 * w_bnd), dtype=pore.dtype)
        pore_expand[w_bnd:w_bnd + ny, w_bnd:w_bnd + nx] = pore

        pore_expand[0, :] = 0
        pore_expand[-1, :] = 0
        pore_expand[:, 0] = 0
        pore_expand[:, -1] = 0

        BW = pore_expand.astype(bool)

        # Skeletonization using porespy
        BW1, _ = ps.networks.skeleton(BW)
        BW2 = BW1[w_bnd:w_bnd + ny, w_bnd:w_bnd + nx]  # midaxis

        # Distance to wall - equivalent to bwdist(~pore)
        D = ndimage.distance_transform_edt(pore.astype(bool))

        # Midaxis to wall (local poresize of midaxis)
        BW3 = D * BW2

        # Index of closest midaxis - equivalent to [~,L] = bwdist(BW2)
        _, L = ndimage.distance_transform_edt(~BW2, return_indices=True)

        # get midaxis indices for all pore locations at once
        midaxis_rows = L[0]
        midaxis_cols = L[1]

        # closest midaxis for each pixel, multiply by 2
        locporesize = BW3[midaxis_rows, midaxis_cols] * 2
        locporesize = locporesize * pore.astype(float)

        # Calculate Knudsen number
        with np.errstate(divide='ignore', invalid='ignore'):
            Kn = lamb / locporesize / Cl
        Kn[~np.isfinite(Kn)] = 0

        # clean pores that are too small
        small_pore_mask = (pore == 1) & (locporesize < 5)
        pore = pore.copy()
        pore[small_pore_mask] = 0

        pore = clean_boundaries(pore)
        diff = np.sum(np.abs(pore_cp.astype(int) - pore.astype(int)))

    return Kn, locporesize, pore, BW3
