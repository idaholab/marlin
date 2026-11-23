"""Gold of coefficient_dt.i with a varying coefficient: a u_t = D Laplacian(u) on the periodic
n x n cell-centred grid of the unit square, with a = 1 + 0.5 sin(2 pi x) sin(2 pi y) and
u(0) = sin(2 pi x) sin(2 pi y), by sparse direct solves of the variable-step BDF2 steps

    A (w0 u^{n+1} + w1 u^n + w2 u^{n-1}) / dt - D L u^{n+1} = 0,   A = diag(a),

with L the periodic 3-point Laplacian, MOOSE's BDF2 weights (sum = dt + dt_old)

    w0 = 1 + dt / sum,  w1 = -sum / dt_old,  w2 = dt^2 / (dt_old sum),

and backward Euler (w0 = 1, w1 = -1, w2 = 0) on the first step. The CSV columns are the
integrals of u and a u (h^2 times the cell sums) and the extremes of u (0 at t = 0, where the
postprocessors have not run). The integral of a u is conserved: the columns of L sum to zero.

Usage: python coefficient_dt.py   (writes gold/coefficient_dt_varying.csv)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D = 0.5
n = 32
times = [0.0, 0.01, 0.02, 0.04, 0.05, 0.065, 0.0725, 0.0875, 0.1175]

h = 1.0 / n
centres = (np.arange(n) + 0.5) * h
# Marlin axis order: the first index is x
x, y = np.meshgrid(centres, centres, indexing="ij")
a = (1 + 0.5 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)).ravel()
u0 = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)).ravel()

# periodic 1D second difference, then the 2D Laplacian as a Kronecker sum
second = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil")
second[0, n - 1] = 1.0
second[n - 1, 0] = 1.0
second = second.tocsr() / h**2
eye = sp.identity(n, format="csr")
laplacian = sp.kron(second, eye) + sp.kron(eye, second)
A = sp.diags(a)

history = [u0]
for step in range(1, len(times)):
    dt = times[step] - times[step - 1]
    if step == 1:
        w0, w1, w2, u_old = 1.0, -1.0, 0.0, np.zeros_like(u0)
    else:
        dt_old = times[step - 1] - times[step - 2]
        total = dt + dt_old
        w0 = 1 + dt / total
        w1 = -total / dt_old
        w2 = dt**2 / (dt_old * total)
        u_old = history[-2]
    matrix = (w0 / dt) * A - D * laplacian
    rhs = -A @ (w1 * history[-1] + w2 * u_old) / dt
    history.append(spla.spsolve(matrix.tocsc(), rhs))

with open("gold/coefficient_dt_varying.csv", "w") as f:
    f.write("time,mass,mass_a,max_u,min_u\n")
    for t, u in zip(times, history):
        row = [h**2 * u.sum(), h**2 * (a * u).sum(), u.max(), u.min()] if t > 0 else [0.0] * 4
        f.write(f"{t:.10g}," + ",".join(f"{v:.15e}" for v in row) + "\n")
