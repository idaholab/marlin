#!/usr/bin/env python3
# *                    DO NOT MODIFY THIS HEADER
# *            Marlin, a Fourier spectral solver for MOOSE
# *
# *            Copyright 2024 Battelle Energy Alliance, LLC
# *                        ALL RIGHTS RESERVED
# *
# *        Licensed under LGPL 2.1, please see LICENSE for details
# *             https://www.gnu.org/licenses/lgpl-2.1.html

"""
Build TorchScript convex-hull surrogates of the Bulatov-Reed-Kumar 5DOF
grain-boundary energy for every pairwise combination of a fixed set of Ni
grain orientations, saved as gb_energy_hull_{2,3}d-style .pt models named
{system}_{name_a}_{name_b}.pt.

Run from this directory so the GB5DOF and GrainBoundaryEnergyHull local
imports resolve:

    python generate_gb_hulls.py
"""

import torch
import GrainBoundaryEnergyHull
from GB5DOF import GB5DOF
from scipy.spatial import ConvexHull
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

system = "Ni"

# Number of GB5DOF evaluations used to construct the reciprocal-space hull.
num_samples = int(1e5)

chunk_size = int(1e3)

dim = 3
device = "cpu"
dtype = torch.float64

beta = 100.0

# Qhull post-merging: merge facets whose centrum lies within this absolute
# distance (in reciprocal-energy space) of a neighboring facet plane. Keeps
# only planes distinguishable at this resolution, bounding the relative
# energy error at roughly this value while cutting the facet count by
# orders of magnitude.
qhull_merge_tolerance = 1.0e-3

qhull_options = f"C-{qhull_merge_tolerance} Qc"


# =============================================================================
# Grain orientations
# =============================================================================

R1 = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)

# Sigma-5-like: 36.86989765 deg about [001]
R2 = torch.tensor(
    [
        [0.8, -0.6, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)

# Sigma-3 twin-like: 60 deg about [111]
R3 = torch.tensor(
    [
        [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
        [2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
        [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
    ],
    dtype=dtype,
    device=device,
)
# Sigma-7-like: 38.21 deg about [111]
R4 = torch.tensor(
    [
        [6.0 / 7.0, 3.0 / 7.0, -2.0 / 7.0],
        [-2.0 / 7.0, 6.0 / 7.0, 3.0 / 7.0],
        [3.0 / 7.0, -2.0 / 7.0, 6.0 / 7.0],
    ],
    dtype=dtype,
    device=device,
)

# Sigma-9-like: 38.94 deg about [110]
R5 = torch.tensor(
    [
        [7.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
        [-4.0 / 9.0, 8.0 / 9.0, -1.0 / 9.0],
        [-4.0 / 9.0, -1.0 / 9.0, 8.0 / 9.0],
    ],
    dtype=dtype,
    device=device,
)

# =============================================================================
# Model
# =============================================================================

model = GB5DOF(
    system=system,
    device=device,
    dtype=dtype,
)


# =============================================================================
# Deterministic sphere sampling
# =============================================================================

def fibonacci_sphere(
    n: int,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """
    Deterministically sample approximately equal-area points on the sphere.

    This avoids the local clustering and holes that occur with random
    Gaussian sampling.
    """
    i = torch.arange(n, device=device, dtype=dtype)

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))

    phi = 2.0 * torch.pi * i / golden_ratio

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)

    return torch.stack([x, y, z], dim=1)


# =============================================================================
# Sample the true GB energy
# =============================================================================

@torch.no_grad()
def sample_inverse_energy_surface(P, Q):
    """
    Sample x = n / gamma(n), which is the reciprocal GB-energy surface.
    """
    normals = fibonacci_sphere(
        num_samples,
        device=device,
        dtype=dtype,
    )

    pts = np.empty(
        (num_samples, dim),
        dtype=np.float64,
    )

    print(f"Sampling {num_samples:,} GB orientations...")

    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)

        normals_chunk = normals[start:end]

        L_chunk = (
            GrainBoundaryEnergyHull
            .lab_rotations_from_normals(normals_chunk)
            .to(device)
        )

        gbe_chunk = model(
            L_chunk @ P,
            L_chunk @ Q,
        )

        pts[start:end] = (
            normals_chunk / gbe_chunk[:, None]
        )[:, :dim].detach().cpu().numpy()

        if start % (10 * chunk_size) == 0:
            print(
                f"  {end:,} / {num_samples:,}"
            )

    if device == "cuda":
        torch.cuda.synchronize()

    return normals, pts


# =============================================================================
# Normalize hull equations
# =============================================================================

def normalize_equations(equations):
    """
    Normalize hull plane equations to

        n . x - h = 0

    where |n| = 1 and h > 0.

    SciPy ConvexHull normally already normalizes the plane normals, but doing
    this explicitly makes all subsequent tolerances meaningful.
    """
    coeffs = equations[:, :dim]
    offsets = equations[:, dim]

    norm = torch.linalg.norm(
        coeffs,
        dim=1,
        keepdim=True,
    )

    coeffs = coeffs / norm
    offsets = offsets / norm[:, 0]

    # For a hull containing the origin, scipy normally gives d < 0.
    # Enforce that convention.
    flip = offsets > 0

    coeffs[flip] *= -1.0
    offsets[flip] *= -1.0

    return torch.cat(
        [
            coeffs,
            offsets[:, None],
        ],
        dim=1,
    )


# =============================================================================
# Build one pair
# =============================================================================

def build_hull_model(
    P,
    Q,
    filename,
):
    print()
    print("=" * 80)
    print(filename)
    print("=" * 80)

    # -------------------------------------------------------------------------
    # True GB5DOF reciprocal-energy surface
    # -------------------------------------------------------------------------

    normals_hull, pts = (
        sample_inverse_energy_surface(
            P,
            Q,
        )
    )

    print(
        "Done sampling. Building merged convex hull..."
    )

    # -------------------------------------------------------------------------
    # Convex hull with Qhull post-merging of near-coplanar facets
    # -------------------------------------------------------------------------

    hull = ConvexHull(
        pts,
        qhull_options=qhull_options,
    )

    print(
        f"Qhull facets ({qhull_options}): "
        f"{len(hull.equations)}"
    )

    equations = torch.tensor(
        hull.equations,
        dtype=dtype,
        device=device,
    )

    equations = normalize_equations(
        equations
    )

    # -------------------------------------------------------------------------
    # Reduced differentiable hull model
    # -------------------------------------------------------------------------

    gb_eh = (
        GrainBoundaryEnergyHull
        .GrainBoundaryEnergyHull(
            equations=equations,
            beta=beta,
        )
    )

    # Evaluate the surrogate on the original normals.
    gb_energies = gb_eh(
        normals_hull
    )

    print(
        "Minimum reduced/smoothed energy =",
        torch.amin(gb_energies).item(),
    )

    # -------------------------------------------------------------------------
    # TorchScript
    # -------------------------------------------------------------------------

    scripted = torch.jit.script(
        gb_eh
    )

    output_filename = f"{system}_{filename}.pt"

    torch.jit.save(
        scripted,
        output_filename,
    )

    print(
        f"Saved TorchScript model: "
        f"{output_filename}"
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    from itertools import combinations

    orientations = {
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "R4": R4,
        "R5": R5,
    }

    for (name_a, Ra), (name_b, Rb) in combinations(
        orientations.items(),
        2,
    ):
        build_hull_model(
            Ra,
            Rb,
            f"{name_a}_{name_b}",
        )
