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
num_samples = int(1e4)

# Number of directions used only to determine how important each hull facet is.
# These are cheap because GB5DOF is NOT evaluated for them.
num_probe_normals = int(1e4)

chunk_size = int(1e3)
probe_chunk_size = int(2e3)

dim = 3
device = "cpu"
dtype = torch.float64

beta = 50.0

# Remove facets whose active region occupies less than this fraction
# of orientation space.
#
# 1e-4 means a facet must control at least ~0.01% of sampled directions.
min_active_fraction = 1.0e-4

# Merge hull planes that are effectively the same plane.
#
# These are deliberately fairly tight. Increase them gradually if the
# reconstructed Wulff shape still contains lots of tiny geometric features.
plane_angle_tol_deg = 0.25
plane_distance_rel_tol = 2.0e-3


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
# Merge almost-identical planes
# =============================================================================

def merge_nearly_identical_planes(
    equations,
    angle_tol_deg=0.25,
    distance_rel_tol=2e-3,
):
    """
    Merge hull planes having nearly identical orientation and distance from
    the origin.

    These usually arise from numerical sampling of what should physically be
    one crystallographic plane.
    """
    eq_np = equations.detach().cpu().numpy()

    normals = eq_np[:, :dim]
    distances = -eq_np[:, dim]

    angle_cos_tol = np.cos(
        np.deg2rad(angle_tol_deg)
    )

    used = np.zeros(
        len(eq_np),
        dtype=bool,
    )

    merged = []
    cluster_sizes = []

    for i in range(len(eq_np)):

        if used[i]:
            continue

        ni = normals[i]
        hi = distances[i]

        dots = normals @ ni

        rel_dist = np.abs(
            distances - hi
        ) / max(abs(hi), 1e-15)

        members = (
            (~used)
            & (dots >= angle_cos_tol)
            & (rel_dist <= distance_rel_tol)
        )

        inds = np.where(members)[0]

        cluster_normals = normals[inds]
        cluster_distances = distances[inds]

        # Average the plane normal, then renormalize.
        n_mean = cluster_normals.mean(axis=0)
        n_mean /= np.linalg.norm(n_mean)

        # Average plane distance from origin.
        h_mean = cluster_distances.mean()

        merged.append(
            np.concatenate(
                [
                    n_mean,
                    [-h_mean],
                ]
            )
        )

        cluster_sizes.append(len(inds))

        used[inds] = True

    merged = np.asarray(
        merged,
        dtype=np.float64,
    )

    print(
        f"Near-plane merge: "
        f"{len(eq_np)} -> {len(merged)} planes"
    )

    if cluster_sizes:
        print(
            f"Largest merged plane cluster: "
            f"{max(cluster_sizes)}"
        )

    return torch.tensor(
        merged,
        dtype=equations.dtype,
        device=equations.device,
    )


# =============================================================================
# Find which facets actually control the energy
# =============================================================================

@torch.no_grad()
def measure_facet_activity(
    equations,
    num_normals,
):
    """
    Measure the fraction of orientation space controlled by each hull plane.

    For

        a_i . x + d_i = 0

    the radial intersection along unit n is

        r_i = -d_i / (a_i . n)

    and therefore the candidate energy is

        gamma_i(n) = (a_i . n) / (-d_i).

    The convex energy is the maximum of these candidates.
    """
    probe_normals = fibonacci_sphere(
        num_normals,
        device=device,
        dtype=dtype,
    )

    coeffs = equations[:, :dim]
    offsets = equations[:, dim]

    n_facets = equations.shape[0]

    counts = torch.zeros(
        n_facets,
        dtype=torch.int64,
        device=device,
    )

    for start in range(
        0,
        num_normals,
        probe_chunk_size,
    ):
        end = min(
            start + probe_chunk_size,
            num_normals,
        )

        n = probe_normals[start:end, :dim]

        denom = n @ coeffs.T

        candidate_energy = (
            denom / (-offsets[None, :])
        )

        # A plane whose outward normal points away from n cannot provide the
        # positive radial intersection in that direction.
        candidate_energy = candidate_energy.masked_fill(
            denom <= 0.0,
            -torch.inf,
        )

        active = candidate_energy.argmax(
            dim=1
        )

        counts += torch.bincount(
            active,
            minlength=n_facets,
        )

    fractions = (
        counts.to(dtype)
        / float(num_normals)
    )

    return counts, fractions


# =============================================================================
# Remove facets with negligible solid-angle support
# =============================================================================

def prune_small_facets(
    equations,
    min_fraction,
):
    counts, fractions = measure_facet_activity(
        equations,
        num_probe_normals,
    )

    keep = fractions >= min_fraction

    # Always retain at least facets that were seen once. This fallback prevents
    # pathological settings of min_fraction from destroying the hull.
    if keep.sum() < 4:
        keep = counts > 0

    kept_equations = equations[keep]

    kept_fractions = fractions[keep]

    order = torch.argsort(
        kept_fractions,
        descending=True,
    )

    kept_equations = kept_equations[order]
    kept_fractions = kept_fractions[order]

    print(
        f"Solid-angle pruning: "
        f"{len(equations)} -> "
        f"{len(kept_equations)} planes"
    )

    print(
        f"Smallest retained active fraction: "
        f"{kept_fractions.min().item():.6e}"
    )

    print(
        f"Largest active fraction: "
        f"{kept_fractions.max().item():.6e}"
    )

    return kept_equations


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
        "Done sampling. Building raw convex hull..."
    )

    hull = ConvexHull(
        pts
    )

    print(
        f"Raw Qhull facets: "
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
    # 1. Remove numerically redundant near-identical planes
    # -------------------------------------------------------------------------

    equations = merge_nearly_identical_planes(
        equations,
        angle_tol_deg=plane_angle_tol_deg,
        distance_rel_tol=plane_distance_rel_tol,
    )

    # -------------------------------------------------------------------------
    # 2. Remove planes active only over negligible solid angle
    # -------------------------------------------------------------------------

    equations = prune_small_facets(
        equations,
        min_fraction=min_active_fraction,
    )

    print(
        f"Final hull plane count: "
        f"{len(equations)}"
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