#!/usr/bin/env python3
"""Reference for bicrystal.i: the backward Euler steps of the two-grain Allen-Cahn system,
each solved to machine precision by Newton's method with the exact Jacobian.

For both order parameters eta_i on the periodic grid (cell centres, spacing h),

    eta_i^{n+1} / dt - L kappa Lap_h eta_i^{n+1} + L dF/deta_i(eta^{n+1}) = eta_i^n / dt,

with the 5-point Laplacian Lap_h and the free energy of bicrystal.i. The Jacobian is sparse
(the two Laplacian blocks plus 2x2 per-cell blocks of L d2F) and is solved directly. Writes the
gold CSV of bicrystal.i (time, area, max_eta1, min_eta2 as the postprocessors compute them; the
postprocessors are not evaluated before the first step, so the t = 0 row is zero).

Run with: python3 bicrystal_newton.py output.csv dt
(gold/bicrystal_anderson.csv with dt = 0.5, gold/bicrystal_picard.csv with dt = 0.25)
"""

import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# the parameters of bicrystal.i (dt from the command line)
L = 0.8545
kappa = 210.6479
m = 1.8724
interface_width = 30.0
r0 = 25.0
n = 24
h = 4.0
num_steps = 5


def laplacian(n, h):
    """5-point periodic Laplacian on an n x n grid, row-major in (x, y)."""
    e = np.ones(n)
    d1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], shape=(n, n), format="lil")
    d1[0, n - 1] = 1
    d1[n - 1, 0] = 1
    d1 = d1.tocsr() / h**2
    eye = sp.identity(n, format="csr")
    return (sp.kron(d1, eye) + sp.kron(eye, d1)).tocsr()


def dF(e1, e2):
    return (
        m * (e1**3 - e1 + 3 * e1 * e2**2),
        m * (e2**3 - e2 + 3 * e2 * e1**2),
    )


def d2F(e1, e2):
    return (
        m * (3 * e1**2 - 1 + 3 * e2**2),
        m * 6 * e1 * e2,
        m * (3 * e2**2 - 1 + 3 * e1**2),
    )


def main():
    out = sys.argv[1]
    dt = float(sys.argv[2])
    x = -n * h / 2 + h / 2 + h * np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    radius = np.sqrt(X**2 + Y**2)
    e1 = (0.5 * (1 - np.tanh(2 * (radius - r0) / interface_width))).ravel()
    e2 = (0.5 * (1 + np.tanh(2 * (radius - r0) / interface_width))).ravel()

    N = n * n
    lap = laplacian(n, h)
    A = sp.identity(N, format="csr") / dt - L * kappa * lap

    rows = []
    for step in range(1, num_steps + 1):
        e1_old, e2_old = e1.copy(), e2.copy()
        for it in range(50):
            f1, f2 = dF(e1, e2)
            R = np.concatenate([A @ e1 + L * f1 - e1_old / dt, A @ e2 + L * f2 - e2_old / dt])
            if np.linalg.norm(R) < 1e-14 * (np.linalg.norm(e1_old) + np.linalg.norm(e2_old)) / dt:
                break
            j11, j12, j22 = d2F(e1, e2)
            J = sp.bmat(
                [
                    [A + L * sp.diags(j11), L * sp.diags(j12)],
                    [L * sp.diags(j12), A + L * sp.diags(j22)],
                ],
                format="csc",
            )
            d = spla.spsolve(J, -R)
            e1 += d[:N]
            e2 += d[N:]
        else:
            raise RuntimeError(f"Newton did not converge in step {step}")
        rows.append((step * dt, e1.copy(), e2.copy()))

    with open(out, "w") as f:
        f.write("time,area,max_eta1,min_eta2\n")
        f.write("0,0,0,0\n")
        for t, a, b in rows:
            # TensorIntegralPostprocessor: the sum times the cell area
            f.write(f"{t:.16e},{a.sum() * h * h:.16e},{a.max():.16e},{b.min():.16e}\n")


if __name__ == "__main__":
    main()
