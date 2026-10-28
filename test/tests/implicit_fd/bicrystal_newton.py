#!/usr/bin/env python3
"""Reference for bicrystal.i: the backward Euler (or variable-step BDF2) steps of the two-grain
Allen-Cahn system, each solved to machine precision by Newton's method with the exact Jacobian.

For both order parameters eta_i on the periodic grid (cell centres, spacing h),

    eta_i^{n+1} / dt - L kappa Lap_h eta_i^{n+1} + L dF/deta_i(eta^{n+1}) = eta_i^n / dt,

with the 5-point Laplacian Lap_h and the free energy of bicrystal.i. With BDF2 the time
derivative is (w0 eta^{n+1} + w1 eta^n + w2 eta^{n-1}) / dt with MOOSE's variable-step weights
(sum = dt + dt_old: w0 = 1 + dt/sum, w1 = -sum/dt_old, w2 = dt^2/(dt_old sum)), backward Euler on
the first step; the steps are dt, dt, 2 dt, dt, 1.5 dt (ratios 0.5 to 2). The Jacobian is sparse
(the two Laplacian blocks plus 2x2 per-cell blocks of L d2F) and is solved directly. Writes the
gold CSV of bicrystal.i (time, area, max_eta1, min_eta2 as the postprocessors compute them; the
postprocessors are not evaluated before the first step, so the t = 0 row is zero).

Run with: python3 bicrystal_newton.py output.csv dt [bdf2]
(gold/bicrystal_anderson.csv with dt = 0.5, gold/bicrystal_picard.csv with dt = 0.25,
gold/bicrystal_bdf2.csv with dt = 0.25 bdf2)
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
    bdf2 = len(sys.argv) > 3 and sys.argv[3] == "bdf2"
    steps = [dt * f for f in (1, 1, 2, 1, 1.5)] if bdf2 else [dt] * num_steps
    x = -n * h / 2 + h / 2 + h * np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    radius = np.sqrt(X**2 + Y**2)
    e1 = (0.5 * (1 - np.tanh(2 * (radius - r0) / interface_width))).ravel()
    e2 = (0.5 * (1 + np.tanh(2 * (radius - r0) / interface_width))).ravel()

    N = n * n
    lap = laplacian(n, h)

    rows = []
    t = 0.0
    e1_older = e2_older = None
    for step in range(1, len(steps) + 1):
        dt = steps[step - 1]
        e1_old, e2_old = e1.copy(), e2.copy()
        # backward Euler (w0 = 1, w1 = -1, w2 = 0), and BDF2 after the first step
        A = sp.identity(N, format="csr") / dt - L * kappa * lap
        r1, r2 = e1_old / dt, e2_old / dt
        if bdf2 and step > 1:
            dt_old = steps[step - 2]
            total = dt + dt_old
            w0, w1, w2 = 1 + dt / total, -total / dt_old, dt**2 / (dt_old * total)
            A = w0 * sp.identity(N, format="csr") / dt - L * kappa * lap
            r1 = -(w1 * e1_old + w2 * e1_older) / dt
            r2 = -(w1 * e2_old + w2 * e2_older) / dt
        for it in range(50):
            f1, f2 = dF(e1, e2)
            R = np.concatenate([A @ e1 + L * f1 - r1, A @ e2 + L * f2 - r2])
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
        e1_older, e2_older = e1_old, e2_old
        t += dt
        rows.append((t, e1.copy(), e2.copy()))

    with open(out, "w") as f:
        f.write("time,area,max_eta1,min_eta2\n")
        f.write("0,0,0,0\n")
        for t, a, b in rows:
            # TensorIntegralPostprocessor: the sum times the cell area
            f.write(f"{t:.16e},{a.sum() * h * h:.16e},{a.max():.16e},{b.min():.16e}\n")


if __name__ == "__main__":
    main()
