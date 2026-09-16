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
Build a small closed-form anisotropic grain-boundary energy model,
sigma(n) = sqrt((sigma_a * n_x)^2 + (sigma_b * n_y)^2), and save it as a
TorchScript module for the single-order-parameter regression test.

GradientVector zero-pads the z component in 2D, so this elliptical form
depends only on the in-plane components of the interface normal.
"""

import torch
import torch.nn as nn


class AnalyticSigma(nn.Module):
    """Elliptical anisotropic GB energy, sigma(n) = |(sigma_a n_x, sigma_b n_y)|."""

    def __init__(self, sigma_a: float, sigma_b: float):
        super().__init__()
        self.register_buffer("sigma_a", torch.tensor(sigma_a, dtype=torch.float64))
        self.register_buffer("sigma_b", torch.tensor(sigma_b, dtype=torch.float64))

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            (self.sigma_a * n[:, 0]) ** 2 + (self.sigma_b * n[:, 1]) ** 2
        )


def main():
    model = AnalyticSigma(sigma_a=1.0, sigma_b=0.7)
    scripted = torch.jit.script(model)

    n = torch.nn.functional.normalize(torch.randn(8, 3, dtype=torch.float64), dim=1)
    n[:, 2] = 0.0
    print("Sample energies:", scripted(n))

    torch.jit.save(scripted, "analytic_sigma.pt")
    print("Saved analytic_sigma.pt")


if __name__ == "__main__":
    main()
