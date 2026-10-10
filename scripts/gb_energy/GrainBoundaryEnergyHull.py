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
Smooth, differentiable, TorchScript-scriptable convex-hull approximation of a
grain-boundary inclination energy surface, plus a helper for building lab
frames from interface normal vectors.
"""

import torch
import torch.nn as nn


def lab_rotations_from_normals(
    normals: torch.Tensor,
) -> torch.Tensor:
    """
    Build orthonormal lab frames [n, t1, t2] for each normal vector.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape [M, 3, 3].
    """
    n = normals

    ref = (
        torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=n.dtype,
            device=n.device,
        )
        .expand_as(n)
        .clone()
    )

    ref[
        torch.abs((n * ref).sum(dim=1)) > 0.99
    ] = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=n.dtype,
        device=n.device,
    )

    t1 = ref - (ref * n).sum(dim=1, keepdim=True) * n
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True)

    t2 = torch.cross(n, t1, dim=1)
    t2 = t2 / torch.linalg.norm(t2, dim=1, keepdim=True)

    return torch.stack([n, t1, t2], dim=1)


class GrainBoundaryEnergyHull(nn.Module):
    """
    Smooth convex-hull approximation of the GB inclination energy surface.

    Differentiable with respect to input normals.

    Supports 2D (circle) and 3D (sphere) inclination surfaces.

    Hull equations are:
        [E, 3] for 2D: (a, b, c)
        [E, 4] for 3D: (a, b, c, d)
    """

    def __init__(
        self,
        equations: torch.Tensor,
        beta: float = 50.0,
        eps: float = 1e-8,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.dtype = GrainBoundaryEnergyHull._resolve_dtype(
            device,
            dtype,
        )

        self._device = device
        self.beta = beta
        self.eps = eps

        self.hull_dim = equations.shape[1] - 1

        self.register_buffer(
            "equations",
            equations,
        )

    def forward(
        self,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        n = normals[:, :self.hull_dim]

        coeffs = self.equations[:, :self.hull_dim]
        offsets = self.equations[:, self.hull_dim]

        denom = n @ coeffs.T

        E_candidate = denom / (-offsets[None, :])

        E_candidate = E_candidate.masked_fill(
            denom <= self.eps,
            -1e6,
        )

        # Softmax-weighted mean.
        #
        # Exponentially accurate near low-energy vertices while providing
        # a smooth differentiable approximation across facet transitions.
        weights = torch.softmax(
            self.beta * E_candidate,
            dim=1,
        )

        return (
            weights * E_candidate
        ).sum(dim=1)

    @staticmethod
    def _resolve_dtype(
        device: torch.device,
        dtype: torch.dtype = None,
    ) -> torch.dtype:
        if dtype is not None:
            return dtype

        return (
            torch.float32
            if device.type == "mps"
            else torch.float64
        )
