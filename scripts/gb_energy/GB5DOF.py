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
Fast, differentiable, TorchScript-compatible implementation of Bulatov's
5-degree-of-freedom (5DOF) grain-boundary energy model.
"""

import torch
import torch.nn as nn


class GB5DOF(nn.Module):
    """
    torch.nn class for fast calculation of Bulatov's 5DOF GB energy model.
    """

    def __init__(
        self, system: str = "Ni", device: torch.device = None, dtype: torch.dtype = None
    ):
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.dtype = GB5DOF._resolve_dtype(device, dtype)
        self._device = device

        # Reused constants
        self.dismax = 0.9999

        self.register_buffer("_half", torch.tensor(0.5, dtype=self.dtype))
        self.register_buffer("_zero", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer(
            "_th3_100", torch.acos(torch.tensor(4 / 5, dtype=self.dtype))
        )
        self.register_buffer(
            "_th5_100", torch.acos(torch.tensor(3 / 5, dtype=self.dtype))
        )
        self.register_buffer(
            "_th6_100", 2 * torch.acos(torch.tensor(5 / 34**0.5, dtype=self.dtype))
        )
        self.register_buffer(
            "_th3_110", torch.acos(torch.tensor(1 / 3, dtype=self.dtype))
        )
        self.register_buffer(
            "_th3_111", torch.acos(torch.tensor(1 / 3, dtype=self.dtype))
        )
        self.register_buffer(
            "_th5_110", torch.acos(torch.tensor(-7 / 11, dtype=self.dtype))
        )
        self.register_buffer("_offset", torch.tensor(1e-5, dtype=self.dtype))

        # ── Material parameters ──────────────────────────────────────────
        par43 = self._make_parvec(system, dtype=self.dtype, device=self._device)
        self.register_buffer("par43", par43)

        sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=self.dtype, device=self._device))
        sqrt_3 = torch.sqrt(torch.tensor(3.0, dtype=self.dtype, device=self._device))

        # ── Axis sets ────────────────────────────────────────────────────
        self.register_buffer(
            "axes_100",
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=self.dtype, device=self._device
            ),
        )
        self.register_buffer(
            "dirs_100",
            torch.tensor(
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=self.dtype, device=self._device
            ),
        )

        self.register_buffer(
            "axes_110",
            torch.tensor(
                [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
                dtype=self.dtype,
                device=self._device,
            )
            / sqrt_2,
        )
        self.register_buffer(
            "dirs_110",
            torch.tensor(
                [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
                dtype=self.dtype,
                device=self._device,
            ),
        )

        self.register_buffer(
            "axes_111",
            torch.tensor(
                [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                dtype=self.dtype,
                device=self._device,
            )
            / sqrt_3,
        )
        self.register_buffer(
            "dirs_111",
            torch.tensor(
                [[1, -1, 0], [1, 1, 0], [1, 1, 0], [1, -1, 0]],
                dtype=self.dtype,
                device=self._device,
            )
            / sqrt_2,
        )

        # ── Precomputed dir2s (axes × dirs) ──────────────────────────────
        self.register_buffer(
            "dir2s_100", torch.linalg.cross(self.axes_100, self.dirs_100)
        )
        self.register_buffer(
            "dir2s_110", torch.linalg.cross(self.axes_110, self.dirs_110)
        )
        self.register_buffer(
            "dir2s_111", torch.linalg.cross(self.axes_111, self.dirs_111)
        )

        # ── Periods ──────────────────────────────────────────────────────
        self.period_100 = torch.pi * self.axes_100.shape[0] / 6  # pi/2
        self.period_110 = torch.pi * self.axes_110.shape[0] / 6  # pi
        self.period_111 = torch.pi * self.axes_111.shape[0] / 6  # 2pi/3

        # ── Symmetry rotation operators ──────────────────────────────────
        self.register_buffer(
            "rotX90",
            torch.tensor(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                dtype=self.dtype,
                device=self._device,
            ),
        )
        self.register_buffer(
            "rotY90",
            torch.tensor(
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                dtype=self.dtype,
                device=self._device,
            ),
        )
        self.register_buffer(
            "rotZ90",
            torch.tensor(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                dtype=self.dtype,
                device=self._device,
            ),
        )
        self.register_buffer(
            "rotZ90m",
            torch.tensor(
                [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                dtype=self.dtype,
                device=self._device,
            ),
        )

    def forward(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        P, Q: (N, 3, 3) orientation matrices
        returns: (N,) GB energies in J/m²
        """
        shared = self._compute_shared(P, Q)

        geom_100 = self._distances_to_set(
            P, shared, self.axes_100, self.dirs_100, self.dir2s_100, self.period_100
        )
        geom_110 = self._distances_to_set(
            P, shared, self.axes_110, self.dirs_110, self.dir2s_110, self.period_110
        )
        geom_111 = self._distances_to_set(
            P, shared, self.axes_111, self.dirs_111, self.dir2s_111, self.period_111
        )

        return self._weighted_mean_energy(geom_100, geom_110, geom_111)

    def _weighted_mean_energy(self, geom_100, geom_110, geom_111):
        eRGB = self.par43[0:1]
        d0100 = self.par43[1:2]
        d0110 = self.par43[2:3]
        d0111 = self.par43[3:4]
        weight100 = self.par43[4:5]
        weight110 = self.par43[5:6]
        weight111 = self.par43[6:7]

        offset = self._offset.item()

        d100 = geom_100[0]
        d110 = geom_110[0]
        d111 = geom_111[0]

        valid100 = geom_100[4]
        valid110 = geom_110[4]
        valid111 = geom_111[4]

        e100 = torch.where(valid100, self._set100(geom_100), torch.zeros_like(d100))
        e110 = torch.where(valid110, self._set110(geom_110), torch.zeros_like(d110))
        e111 = torch.where(valid111, self._set111(geom_111), torch.zeros_like(d111))

        s100 = torch.where(
            d100 > d0100,
            torch.ones_like(d100),
            torch.where(
                d100 >= offset * d0100,
                torch.sin(torch.pi * d100 / (2.0 * d0100)),
                torch.full_like(d100, offset * torch.pi / 2.0),
            ),
        )
        w100 = torch.where(
            valid100,
            (1.0 / (s100 * (1.0 - 0.5 * torch.log(s100))) - 1.0) * weight100,
            torch.zeros_like(s100),
        )

        s110 = torch.where(
            d110 > d0110,
            torch.ones_like(d110),
            torch.where(
                d110 >= offset * d0110,
                torch.sin(torch.pi * d110 / (2.0 * d0110)),
                torch.full_like(d110, offset * torch.pi / 2.0),
            ),
        )
        w110 = torch.where(
            valid110,
            (1.0 / (s110 * (1.0 - 0.5 * torch.log(s110))) - 1.0) * weight110,
            torch.zeros_like(s110),
        )

        s111 = torch.where(
            d111 > d0111,
            torch.ones_like(d111),
            torch.where(
                d111 >= offset * d0111,
                torch.sin(torch.pi * d111 / (2.0 * d0111)),
                torch.full_like(d111, offset * torch.pi / 2.0),
            ),
        )
        w111 = torch.where(
            valid111,
            (1.0 / (s111 * (1.0 - 0.5 * torch.log(s111))) - 1.0) * weight111,
            torch.zeros_like(s111),
        )

        num = (
            torch.sum(e100 * w100, dim=1)
            + torch.sum(e110 * w110, dim=1)
            + torch.sum(e111 * w111, dim=1)
            + 1.0
        )

        den = (
            torch.sum(w100, dim=1)
            + torch.sum(w110, dim=1)
            + torch.sum(w111, dim=1)
            + 1.0
        )
        en = eRGB * num / den
        return en

    def _twist100(self, geom100):
        a = self.par43[9:10]
        b = self.par43[9:10] * self.par43[10:11]

        period = torch.pi / 2
        ksi = torch.remainder(torch.abs(geom100[1]), period)

        ksi_periodic = torch.where(ksi <= period / 2, ksi, period - ksi)

        sins = torch.sin(2 * ksi_periodic)
        sins_for_log = sins.clamp_min(torch.finfo(sins.dtype).tiny)
        xlogx = sins * torch.log(sins_for_log)
        en = a * sins - b * xlogx
        return en

    def _atgb100(self, geom100):
        pwr = self.par43[11:12]
        period = torch.pi / 2

        eta = geom100[2]
        ksi = geom100[1]

        en1 = self._stgb100(ksi)
        en2 = self._stgb100(period - ksi)

        select = en1 >= en2

        en_hi = en1 - (en1 - en2) * (eta / period) ** pwr
        en_lo = en2 - (en2 - en1) * (1.0 - eta / period) ** pwr

        en = torch.where(select, en_hi, en_lo)
        return en

    def _stgb100(self, ksi):
        en2 = self.par43[12:13]  # peak before first Sigma5
        en3 = self.par43[13:14]  # first Sigma5
        en4 = self.par43[14:15]  # peak between Sigma5's
        en5 = self.par43[15:16]  # second Sigma5
        en6 = self.par43[16:17]  # Sigma17

        th2 = self.par43[17:18]  # position of peak before first Sigma5
        th4 = self.par43[18:19]  # position of peak between Sigma5's

        th6 = self._th6_100  # Sigma17 rotation angle
        # rsw shape factor.  In previous versions, these were allowed
        a12 = self._half
        a23 = a12  # to vary, however there were too few vicinal boundaries in the
        a34 = a12  # ensemble to constrain them.  We found that forcing the great
        a45 = a12  # majority of them to be 0.5 helped to constrain the fit and
        a56 = a12  # produced reasonable results.  This holds true for most of the
        a67 = a12  # rsw shape factors throughout this code.

        # Sigma1 at left end
        en1 = self._zero
        # Sigma1 at right end
        en7 = self._zero

        # Sigma1 at left end
        th1 = self._zero
        th3 = self._th3_100  # first Sigma5
        th5 = self._th5_100  # second Sigma5
        th7 = torch.pi / 2  # Sigma1 at right end

        # Directly calculate all the RSW functions first
        f1 = en1 + (en2 - en1) * self._rsw(ksi, th1, th2, a12)
        f2 = en3 + (en2 - en3) * self._rsw(ksi, th3, th2, a23)
        f3 = en3 + (en4 - en3) * self._rsw(ksi, th3, th4, a34)
        f4 = en5 + (en4 - en5) * self._rsw(ksi, th5, th4, a45)
        f5 = en6 + (en5 - en6) * self._rsw(ksi, th6, th5, a56)
        f6 = en7 + (en6 - en7) * self._rsw(ksi, th7, th6, a67)

        en = torch.where(
            ksi <= th2,
            f1,
            torch.where(
                ksi <= th3,
                f2,
                torch.where(
                    ksi <= th4,
                    f3,
                    torch.where(ksi <= th5, f4, torch.where(ksi <= th6, f5, f6)),
                ),
            ),
        )
        return en

    def _twists110(self, geom110):
        ksi = geom110[1]
        th1 = self.par43[21:22]  # 110 twist peak position

        en1 = self.par43[22:23]  # 110 twist energy peak value
        # Sigma3 energy (110 twist, so not a coherent twin)
        en2 = self.par43[23:24]
        en3 = self.par43[24:25]  # energy at the symmetry point

        a01 = a12 = a23 = self._half

        th2 = self._th3_110  # Sigma3
        th3 = torch.pi / 2  # 110 90-degree boundary is semi-special, although not a CSL

        period = torch.pi  # the twist period

        th_unthreshed = torch.remainder(torch.abs(ksi), period)  # rotation symmetry
        th = torch.where(
            th_unthreshed > period / 2, period - th_unthreshed, th_unthreshed
        )

        f1 = en1 * self._rsw(
            th, torch.tensor(0, dtype=self.dtype, device=self._device), th1, a01
        )
        f2 = en2 + (en1 - en2) * self._rsw(th, th2, th1, a12)
        f3 = en3 + (en2 - en3) * self._rsw(th, th3, th2, a23)

        en = torch.where(th <= th1, f1, torch.where(th <= th2, f2, f3))
        return en

    def _atgbs110(self, geom110):

        a = self.par43[25:26]  # 110 atgb interpolation rsw shape factor
        eta = geom110[2]
        ksi = geom110[1]

        period = torch.pi

        en1 = self._stgbs110(ksi)
        en2 = self._stgbs110(period - ksi)

        select = en1 >= en2

        rsw_eta = self._rsw(eta, self._zero, torch.pi, a)

        en = torch.where(
            select, en2 + (en1 - en2) * rsw_eta, en1 + (en2 - en1) * rsw_eta
        )
        return en

    def _stgbs110(self, th):
        en2 = self.par43[26:27]  # peak between Sigma1 and Sigma3
        # Coherent Sigma3 twin relative energy;
        # one of the more important element-dependent parameters
        en3 = self.par43[27:28]
        en4 = self.par43[28:29]  # energy peak between Sigma3 and Sigma11
        en5 = self.par43[29:30]  # Sigma11 energy
        en6 = self.par43[30:31]  # energy peak between Sigma11 and Sigma1

        th2 = self.par43[31:32]  # peak between Sigma1 and Sigma3
        th4 = self.par43[32:33]  # peak between Sigma3 and Sigma11
        th6 = self.par43[33:34]  # peak between Sigma11 and higher Sigma1

        a12 = a23 = a34 = a45 = a56 = a67 = self._half

        en1 = self._zero
        en7 = self._zero

        th1 = self._zero
        th3 = self._th3_110  # Sigma3
        th5 = self._th5_110  # Sigma11
        th7 = torch.pi

        th = torch.pi - th  # This is a legacy of an earlier (ksi,eta) mapping

        f1 = en1 + (en2 - en1) * self._rsw(th, th1, th2, a12)
        f2 = en3 + (en2 - en3) * self._rsw(th, th3, th2, a23)
        f3 = en3 + (en4 - en3) * self._rsw(th, th3, th4, a34)
        f4 = en5 + (en4 - en5) * self._rsw(th, th5, th4, a45)
        f5 = en5 + (en6 - en5) * self._rsw(th, th5, th6, a56)
        f6 = en7 + (en6 - en7) * self._rsw(th, th7, th6, a67)

        en = torch.where(
            th <= th2,
            f1,
            torch.where(
                th <= th3,
                f2,
                torch.where(
                    th <= th4,
                    f3,
                    torch.where(th <= th5, f4, torch.where(th <= th6, f5, f6)),
                ),
            ),
        )
        return en

    def _twists111(self, geom111):
        ksi = geom111[1]
        thd = self.par43[36:37]
        enm = self.par43[37:38]
        en2 = self.par43[27:28]
        a1 = self.par43[35:36]  # 111 twist rsw shape parameter
        a2 = a1

        theta = torch.where(ksi > torch.pi / 3, 2 * torch.pi / 3 - ksi, ksi)

        select = theta <= thd
        f1 = enm * self._rsw(theta, self._zero, thd, a1)
        f2 = en2 + (enm - en2) * self._rsw(theta, torch.pi / 3, thd, a2)

        en = torch.where(select, f1, f2)
        return en

    def _atgbs111(self, geom111):
        """
        % This function is a fit to the energies of all 111-tilt boundaries
        """

        eta = geom111[2]
        ksi = geom111[1]

        # There's an additional symmetry in 111 atgbs that doesn't exist in 100 or
        # 110 atgbs.  This is because a rotation about [111] equal to half the period
        # (i.e. 60 degrees) is equivalent to a mirror reflection in the (111)
        # plane.  Both are Sigma3 operations.  The same is not true of the
        # 45-degree [100] or the 90-degree [110] rotation.
        # The following two lines account for this extra symmetry.

        ksi_sym = torch.where(ksi > torch.pi / 3, 2 * torch.pi / 3 - ksi, ksi)
        eta_sym = torch.where(eta > torch.pi / 3, 2 * torch.pi / 3 - eta, eta)

        #     Below the following value of ksi, we ignore the eta dependence.  This is
        #     because there's little evidence that it actually varies.  Above this
        #     value, we interpolate on an rsw function that follows the Sigma3 line,
        #     which is also a line of symmetry for the function.
        ksim = self.par43[38:39]  # 111 atgb ksi break

        enmax = self.par43[39:40]
        enmin = self.par43[40:41]
        encnt = self.par43[41:42]

        a1 = a2 = self._half

        # eta scaling parameter for 111 atgb rsw function on Sigma3 line
        etascale = self.par43[42:43]

        #  This rsw function is unusual in that the change in shape of the
        #  function is much better captured by changing the angular scale rather
        #  than changing the dimensionless shape factor.

        select = ksi_sym <= ksim

        f1 = enmax * self._rsw(
            ksi_sym, torch.tensor(0, dtype=self.dtype, device=self._device), ksim, a1
        )
        chi = enmin + (encnt - enmin) * self._rsw(
            eta_sym.clamp(max=torch.pi / (2 * etascale)),
            0,
            torch.pi / (2 * etascale),
            torch.tensor(0.5, dtype=self.dtype, device=self._device),
        )
        f2 = chi + (enmax - chi) * self._rsw(ksi_sym, torch.pi / 3, ksim, a2)
        en = torch.where(select, f1, f2)
        return en

    def _set100(self, geom100):
        # Calculate the dimensionless contribution to the boundary based on the
        # nearby <100> rotations.  Meant to be called by weightedmeanenergy.m, but
        # also can be a stand-alone function for purposes of plotting cross
        # sections through the function.
        # Input variables geom100 and self.par43 are as generated by distances_to_set.m
        # and makeparvec.m.  See comments in those functions for more information.

        pwr1 = self.par43[7:8]
        pwr2 = self.par43[8:9]

        phi = geom100[3]

        entwist = self._twist100(geom100)
        entilt = self._atgb100(geom100)

        x = phi / (torch.pi / 2)
        en = entwist * (1 - x) ** pwr1 + entilt * x**pwr2
        return en

    def _set110(self, geom110):
        # Dimensionless contribution to energy from <110> rotations
        # Very similar to set100; see comments therein for general information.
        # Comments in this file will be limited to 110-specific information.

        pwr1 = self.par43[19:20]  # 110 tilt/twist mix power law:  Twist
        pwr2 = self.par43[20:21]  # 110 tilt/twist mix power law:  Tilt

        entwist = self._twists110(geom110)
        entilt = self._atgbs110(geom110)

        phi = geom110[3]
        x = phi / (torch.pi / 2)
        en = entwist * (1 - x) ** pwr1 + entilt * x**pwr2
        return en

    def _set111(self, geom111):
        # Dimensionless contribution to energy from <111> rotations
        # Very similar to set100; see comments therein for general information.
        # Comments in this file will be limited to 111-specific information.

        a = self.par43[34:35]  # linear part of 111 tilt/twist interpolation
        b = a - 1  # Ensures correct value at x = 1.

        phi = geom111[3]

        entwist = self._twists111(geom111)
        entilt = self._atgbs111(geom111)
        x = phi / (torch.pi / 2)

        #   This one fit well enough with a simple one-parameter parabola that the
        #   more complicated power laws in the other sets weren't needed.
        en = entwist + (entilt - entwist) * (a * x - b * x**2)
        return en

    def _rsw(
        self,
        theta: torch.Tensor,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized Read-Shockley-Wolf function.

        Computes:
            en = sin(x) - a * sin(x) * log(sin(x))
        where
            x = (theta - theta1) / (theta2 - theta1) * pi/2

        Works with broadcastable tensor/scalar inputs.
        """
        dtheta = theta2 - theta1
        x = (theta - theta1) / dtheta * (torch.pi / 2)

        s = torch.sin(x).clamp(min=1e-12)

        # MATLAB behavior: xlogx = 0 when s -> 0
        # Clamp only for the log; keep the original s in front.
        s_for_log = s.clamp_min(torch.finfo(s.dtype).tiny)
        xlogx = s * torch.log(s_for_log)

        return s - a * xlogx

    def _distances_to_set(
        self,
        P: torch.Tensor,
        shared: tuple,
        axes: torch.Tensor,
        dirs: torch.Tensor,
        dir2s: torch.Tensor,
        period: torch.Tensor,
    ) -> torch.Tensor:

        V, axi, psi = shared
        dotp = (axi[:, :, None, :] * axes[None, None, :, :]).sum(dim=-1)

        # Compute rotational distance from boundary P/Q to the rotation set "i"
        # This formula produces 2*sin(delta/2), where delta is the angle of
        # closest approach.
        dis = (
            2.0
            * torch.sqrt(torch.abs(1.0 - dotp * dotp).clamp(min=0.0))
            * torch.sin(psi[:, :, None] / 2.0)
        )

        psi_half = psi[:, :, None] / 2.0
        theta = 2.0 * torch.atan2(dotp * torch.sin(psi_half), torch.cos(psi_half))

        # Compute the normal of the best-fitting GB in grain 1
        n1 = P[:, 0, :]  # [N,3]
        n2 = V[:, :, 0, :]  # [N,24,3]

        # RA is the rotation about ax that most closely approximates R
        half_theta = theta / 2.0  # [N,24,A]
        c = torch.cos(half_theta)  # [N,24,A]
        s = torch.sin(half_theta)  # [N,24,A]
        ax = axes[None, None, :, :]  # [1,1,A,3]

        qRA = torch.cat(
            [
                c[..., None],  # [N,24,A,1]
                s[..., None] * ax,  # [N,24,A,3]
            ],
            dim=-1,
        )  # [N,24,A,4]

        RA = self.quat2mat(qRA)  # [N,24,A,3,3]

        n2_col = n2[:, :, None, :, None]  # [N,24,1,3,1]
        RA_T_n2 = (RA.transpose(-1, -2) @ n2_col).squeeze(-1)  # [N,24,A,3]

        n1_exp = n1[:, None, None, :]  # [N,1,1,3]
        m1 = n1_exp + RA_T_n2  # [N,24,A,3]

        m1_norm = torch.linalg.norm(m1, dim=-1, keepdim=True)
        m1 = m1 / m1_norm

        # Ensure any points that are near singular are discarded
        dis[m1_norm.squeeze(-1) < 1e-6] = 2 * self.dismax

        m2 = torch.einsum("nkaij,nkaj->nkai", RA, m1)

        # dot product m1 · ax
        dot_m1_ax = (m1 * axes[None, None, :, :]).sum(dim=-1)

        # clamp for numerical safety
        dot_m1_ax = dot_m1_ax.abs().clamp(-1.0, 1.0)

        phi = torch.acos(dot_m1_ax)  # [N,24,A]

        ax_exp = axes[None, None, :, :]  # [1,1,A,3]
        dir_exp = dirs[None, None, :, :]  # [1,1,A,3]
        dir2_exp = dir2s[None, None, :, :]  # [1,1,A,3]

        twist_mask = (m1 * ax_exp).sum(dim=-1).abs() > 0.9999  # [N,24,A]

        theta1_twist = -theta / 2.0
        theta2_twist = theta / 2.0

        theta1_general = torch.atan2(
            (m1 * dir2_exp).sum(dim=-1),
            (m1 * dir_exp).sum(dim=-1),
        )

        theta2_general = torch.atan2(
            (m2 * dir2_exp).sum(dim=-1),
            (m2 * dir_exp).sum(dim=-1),
        )

        theta1 = torch.where(twist_mask, theta1_twist, theta1_general)
        theta2 = torch.where(twist_mask, theta2_twist, theta2_general)

        theta2 = theta2 - torch.round(theta2 / period) * period
        theta1 = theta1 - torch.round(theta1 / period) * period

        theta2 += torch.where(
            torch.abs(theta2 + period / 2) < 1e-6, period, torch.zeros_like(theta2)
        )
        theta1 += torch.where(
            torch.abs(theta1 + period / 2) < 1e-6, period, torch.zeros_like(theta1)
        )

        ksi = torch.abs(theta2 - theta1)
        eta = torch.abs(theta2 + theta1)

        N, K, A = dis.shape
        M = K * A

        dis_flat = dis.reshape(N, M)
        ksi_flat = ksi.reshape(N, M)
        eta_flat = eta.reshape(N, M)
        phi_flat = phi.reshape(N, M)

        # Checks for both point uniqueness and distance < threshold
        sorted_set = self._fastsort_unique_mask(
            dis_flat, ksi_flat, eta_flat, phi_flat, self.dismax
        )
        valid_final, dis_final, ksi_final, eta_final, phi_final = sorted_set

        geom = (dis_final, ksi_final, eta_final, phi_final, valid_final)
        return geom

    def _fastsort_unique_mask(
        self,
        dis_sorted: torch.Tensor,
        ksi_sorted: torch.Tensor,
        eta_sorted: torch.Tensor,
        phi_sorted: torch.Tensor,
        dismax: float,
    ):
        """
        Build a validity mask that keeps only the first occurrence of each
        rounded (dis, ksi, eta, phi) tuple with dis < dismax.

        Inputs:
            dis_sorted, ksi_sorted, eta_sorted, phi_sorted: [N, M]
        Returns:
            valid:      [N, M] boolean mask in lexicographically sorted order
            perm:       [N, M] permutation indices used for lexicographic sorting
            dis_lx:     [N, M]
            ksi_lx:     [N, M]
            eta_lx:     [N, M]
            phi_lx:     [N, M]
            unique_sel: [N, M]
        """
        N, M = dis_sorted.shape

        dis_r = torch.round(dis_sorted, decimals=6)
        ksi_r = torch.round(ksi_sorted, decimals=6)
        eta_r = torch.round(eta_sorted, decimals=6)
        phi_r = torch.round(phi_sorted, decimals=6)

        # Current indices
        idx = torch.arange(M, device=dis_sorted.device).expand(N, M)

        for key in (dis_r, ksi_r, eta_r, phi_r):
            order = torch.argsort(key.gather(1, idx), dim=1, stable=True)
            idx = idx.gather(1, order)

        # Gather quantized values in sorted order
        sd = dis_r.gather(1, idx)
        sk = ksi_r.gather(1, idx)
        se = eta_r.gather(1, idx)
        sp = phi_r.gather(1, idx)

        same_as_prev = (
            (sd[:, 1:] == sd[:, :-1])
            & (sk[:, 1:] == sk[:, :-1])
            & (se[:, 1:] == se[:, :-1])
            & (sp[:, 1:] == sp[:, :-1])
        )

        unique_sorted = torch.ones((N, M), dtype=torch.bool, device=dis_sorted.device)
        unique_sorted[:, 1:] = ~same_as_prev

        dis_perm = dis_sorted.gather(1, idx)
        valid_sorted = unique_sorted & (dis_perm < dismax)

        # Scatter masks back to original order
        valid = torch.zeros((N, M), dtype=torch.bool, device=dis_sorted.device)
        valid.scatter_(1, idx, valid_sorted)

        return valid, dis_sorted, ksi_sorted, eta_sorted, phi_sorted

    def _compute_shared(self, P: torch.Tensor, Q: torch.Tensor):
        """
        Compute the 24 symmetry variants and quaternion decomposition.
        Called once per forward pass, results shared across all three
        axis set calculations.

        Returns: (V, axi, psi)
        """

        # Get shape of input rotation tensors
        N, a, b = Q.shape
        V = torch.zeros(N, 24, a, b, dtype=Q.dtype, device=Q.device)

        V[:, 0] = Q
        V[:, 1] = Q @ self.rotX90
        V[:, 2] = V[:, 1] @ self.rotX90
        V[:, 3] = V[:, 2] @ self.rotX90

        for j in range(12):  # Rotate three times around Y by +90 degrees
            V[:, j + 4] = V[:, j] @ self.rotY90

        for j in range(4):
            # Rotate three times around Z by +90 degrees
            V[:, j + 16] = V[:, j] @ self.rotZ90
            # Rotate three times around Z by -90 degrees
            V[:, j + 20] = V[:, j] @ self.rotZ90m

        P_exp = P[:, None, :, :]  # [N, 1, 3, 3]
        R = V.transpose(-1, -2) @ P_exp

        q = self.mat2quat(R)

        axi = q[..., 1:4] / torch.linalg.norm(
            q[
                ...,
                1:4,
            ],
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-12)
        psi = 2.0 * torch.acos(q[..., 0].clamp(-1.0, 1.0))

        return V, axi, psi

    def mat2quat(self, R: torch.Tensor) -> torch.Tensor:
        orig_shape = R.shape[:-2]
        R = R.reshape(-1, 3, 3)

        m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
        trace = m00 + m11 + m22

        # Compute all 4 cases unconditionally
        # Case 1: trace > 0
        s1 = torch.sqrt((1.0 + trace).clamp(min=1e-12)) * 2.0
        q1 = torch.stack(
            [
                0.25 * s1,
                (R[:, 2, 1] - R[:, 1, 2]) / s1,
                (R[:, 0, 2] - R[:, 2, 0]) / s1,
                (R[:, 1, 0] - R[:, 0, 1]) / s1,
            ],
            dim=-1,
        )

        # Case 2: m00 largest
        s2 = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=1e-12)) * 2.0
        q2 = torch.stack(
            [
                (R[:, 2, 1] - R[:, 1, 2]) / s2,
                0.25 * s2,
                (R[:, 0, 1] + R[:, 1, 0]) / s2,
                (R[:, 0, 2] + R[:, 2, 0]) / s2,
            ],
            dim=-1,
        )

        # Case 3: m11 largest
        s3 = torch.sqrt((1.0 + m11 - m00 - m22).clamp(min=1e-12)) * 2.0
        q3 = torch.stack(
            [
                (R[:, 0, 2] - R[:, 2, 0]) / s3,
                (R[:, 0, 1] + R[:, 1, 0]) / s3,
                0.25 * s3,
                (R[:, 1, 2] + R[:, 2, 1]) / s3,
            ],
            dim=-1,
        )

        # Case 4: m22 largest
        s4 = torch.sqrt((1.0 + m22 - m00 - m11).clamp(min=1e-12)) * 2.0
        q4 = torch.stack(
            [
                (R[:, 1, 0] - R[:, 0, 1]) / s4,
                (R[:, 0, 2] + R[:, 2, 0]) / s4,
                (R[:, 1, 2] + R[:, 2, 1]) / s4,
                0.25 * s4,
            ],
            dim=-1,
        )

        # Select correct case with torch.where — no branching
        c1 = trace > 0
        c2 = (~c1) & (m00 > m11) & (m00 > m22)
        c3 = (~c1) & (~c2) & (m11 > m22)
        c4 = ~(c1 | c2 | c3)

        q = q1 * c1[:, None] + q2 * c2[:, None] + q3 * c3[:, None] + q4 * c4[:, None]

        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
        return q.reshape(*orig_shape, 4)

    def quat2mat(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion(s) to rotation matrix/matrices.

        Input:
            q: [..., 4] where q = [w, x, y, z]

        Output:
            R: [..., 3, 3]

        Works with arbitrary batch dimensions.
        """

        if q.shape[-1] != 4:
            raise ValueError(f"Input must have shape [...,4], got {q.shape}")

        # Normalize quaternion (important for stability)
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)

        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]

        # Precompute products
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z

        wx = w * x
        wy = w * y
        wz = w * z

        xy = x * y
        xz = x * z
        yz = y * z

        # Allocate output
        R = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)

        # Fill matrix
        R[..., 0, 0] = ww + xx - yy - zz
        R[..., 0, 1] = 2 * (xy - wz)
        R[..., 0, 2] = 2 * (xz + wy)

        R[..., 1, 0] = 2 * (xy + wz)
        R[..., 1, 1] = ww - xx + yy - zz
        R[..., 1, 2] = 2 * (yz - wx)

        R[..., 2, 0] = 2 * (xz - wy)
        R[..., 2, 1] = 2 * (yz + wx)
        R[..., 2, 2] = ww - xx - yy + zz

        return R

    @staticmethod
    def _resolve_dtype(device: torch.device, dtype: torch.dtype = None) -> torch.dtype:
        """
        Resolve dtype based on device if not explicitly specified.
        - CPU / CUDA → float64 (default, full precision)
        - MPS       → float32 (MPS does not support float64)
        """
        if dtype is not None:
            return dtype
        if device.type == "mps":
            return torch.float32
        return torch.float64

    @staticmethod
    def _make_parvec(
        system: str, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Build the 43-parameter vector for a given material system.
        Separated as a static method so it can also be used to export
        parameter files independently of the model.
        """

        if system == "Ni":
            eRGB, AlCuparameter = 1.44532834613925, 0.767911805073948
        elif system == "Al":
            eRGB, AlCuparameter = 0.547128733614891, 0.0
        elif system == "Au":
            eRGB, AlCuparameter = 0.529912885175204, 0.784289766313152
        elif system == "Cu":
            eRGB, AlCuparameter = 1.03669431227427, 1.0
        else:
            raise ValueError(
                f"Undefined system: {system}, must be 'Ni','Al','Au', or 'Cu'."
            )

        par42Al = torch.tensor(
            [
                0.405204179289160,
                0.738862004021890,
                0.351631012630026,
                2.40065811939667,
                1.34694439281655,
                0.352260396651516,
                0.602137375062785,
                1.58082498976078,
                0.596442399566661,
                1.30981422643602,
                3.21443408257354,
                0.893016409093743,
                0.835332505166333,
                0.933176738717594,
                0.896076948651935,
                0.775053293192055,
                0.391719619979054,
                0.782601780600192,
                0.678572601273508,
                1.14716256515278,
                0.529386201144101,
                0.909044736601838,
                0.664018011430602,
                0.597206897283586,
                0.200371750006251,
                0.826325891814124,
                0.111228512469435,
                0.664039563157148,
                0.241537262980083,
                0.736315075146365,
                0.514591177241156,
                1.73804335876546,
                3.04687038671309,
                1.48989831680317,
                0.664965104218438,
                0.495035051289975,
                0.495402996460658,
                0.468878130180681,
                0.836548944799803,
                0.619285521065571,
                0.844685390948170,
                1.02295427618256,
            ],
            dtype=dtype,
            device=device,
        )
        par42Cu = torch.tensor(
            [
                0.405204179289160,
                0.738862004021890,
                0.351631012630026,
                2.40065811939667,
                1.34694439281655,
                3.37892632736175,
                0.602137375062785,
                1.58082498976078,
                0.710489498577995,
                0.737834049784765,
                3.21443408257354,
                0.893016409093743,
                0.835332505166333,
                0.933176738717594,
                0.896076948651935,
                0.775053293192055,
                0.509781056492307,
                0.782601780600192,
                0.762160812499734,
                1.10473084066580,
                0.529386201144101,
                0.909044736601838,
                0.664018011430602,
                0.597206897283586,
                0.200371750006251,
                0.826325891814124,
                0.0226010533470218,
                0.664039563157148,
                0.297920289861751,
                0.666383447163744,
                0.514591177241156,
                1.73804335876546,
                2.69805148576400,
                1.95956771207484,
                0.948894352912787,
                0.495035051289975,
                0.301975031994664,
                0.574050577702240,
                0.836548944799803,
                0.619285521065571,
                0.844685390948170,
                0.0491040633104212,
            ],
            dtype=dtype,
            device=device,
        )

        par43 = torch.hstack(
            [
                torch.tensor([eRGB], dtype=dtype, device=device),
                par42Al + AlCuparameter * (par42Cu - par42Al),
            ]
        )
        return par43


if __name__ == "__main__":
    from time import time

    N = 5000

    P = torch.tensor(
        [[2.0, 2.0, 2.0], [1.0, -1.0, 0.0], [1.0, 1.0, -2.0]], dtype=torch.float64
    )

    Q = torch.tensor(
        [[2.0, 2.0, 2.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 2.0]], dtype=torch.float64
    )

    P = P / P.norm(dim=1, keepdim=True)
    Q = Q / Q.norm(dim=1, keepdim=True)

    P = P.unsqueeze(0).repeat(N, 1, 1)
    Q = Q.unsqueeze(0).repeat(N, 1, 1)

    # ── Eager model ──────────────────────────────────────────────────────
    model = GB5DOF(system="Ni", device="cpu")

    # Warmup
    _ = model(P, Q)

    t0 = time()
    for _ in range(10):
        out_eager = model(P, Q)
    t_eager = (time() - t0) / 10
    print(f"Eager:    {t_eager * 1000:.2f} ms  |  result: {out_eager[:3]}")

    # ── Compiled model ───────────────────────────────────────────────────
    torch._dynamo.config.capture_scalar_outputs = True
    compiled = torch.compile(model)

    # First call triggers compilation — time separately
    t_compile_start = time()
    _ = compiled(P, Q)
    t_compile = time() - t_compile_start
    print(f"Compile time: {t_compile:.2f} s")

    # Warmup compiled
    for _ in range(3):
        _ = compiled(P, Q)

    t0 = time()
    for _ in range(10):
        out_compiled = compiled(P, Q)
    t_compiled = (time() - t0) / 10
    print(f"Compiled: {t_compiled * 1000:.2f} ms  |  result: {out_compiled[:3]}")

    # ── Correctness check ────────────────────────────────────────────────
    print(f"\nMax diff: {(out_eager - out_compiled).abs().max().item():.2e}")
    assert torch.allclose(out_eager, out_compiled, atol=1e-10), "Outputs differ!"
    print(f"Speedup: {t_eager / t_compiled:.2f}x")
