"""Gold of sbm_inactive.i: smoothed boundary diffusion psi u_t = div(psi D grad u) on the
periodic n x n cell-centred grid of the unit square, by sparse direct solves of the
variable-step BDF2 steps on the active cells alone.

The discretisation is that of CoefficientTimeDerivativeOperator and FiniteDifferenceDivGrad
(face_mean = arithmetic, factor = -1):

    psi_i (w0 u_i^{n+1} + w1 u_i^n + w2 u_i^{n-1}) / dt
        - sum_faces (psi D)_face (u_j^{n+1} - u_i^{n+1}) / h^2 = 0,

with (psi D)_face the mean of the two cells' psi D, MOOSE's BDF2 weights (sum = dt + dt_old)

    w0 = 1 + dt / sum,  w1 = -sum / dt_old,  w2 = dt^2 / (dt_old sum),

and backward Euler (w0 = 1, w1 = -1, w2 = 0) on the first step. A cell with a zero diagonal
(psi = 0 on the cell and on its faces) has a zero row and column: it is left out of the solve
and keeps u^0. The CSV columns are the integrals of u, psi u and psi u^2 (h^2 times the cell
sums), the extremes of u and held_change, 0 (all 0 at t = 0, where the postprocessors have
not run).

Usage: python sbm_inactive.py   (writes gold/sbm_inactive.csv)
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
r = np.abs(x - 0.5)
psi = np.where(r < 0.2, 1.0, np.where(r < 0.3, 0.5 * (1 + np.cos(np.pi * (r - 0.2) / 0.1)), 0.0))
u0 = 1 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.25 * np.cos(4 * np.pi * y)

# -div(psi D grad u) with arithmetic face means, periodic on both axes
psi_D = D * psi
index = np.arange(n * n).reshape(n, n)
rows, cols, vals = [], [], []
for axis in range(2):
    neighbour = np.roll(index, -1, axis=axis)
    face = 0.5 * (psi_D + np.roll(psi_D, -1, axis=axis)) / h**2
    for a, b in ((index, neighbour), (neighbour, index)):
        rows += [a.ravel(), a.ravel()]
        cols += [a.ravel(), b.ravel()]
        vals += [face.ravel(), -face.ravel()]
L = sp.csr_matrix(
    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)
)
psi_flat = psi.ravel()

# the active cells: a positive diagonal of the operator (the time derivative or a face)
active = psi_flat + L.diagonal() > 0
L_active = L[active][:, active]

history = [u0.ravel()]
for step in range(1, len(times)):
    dt = times[step] - times[step - 1]
    if step == 1:
        w0, w1, w2, u_old = 1.0, -1.0, 0.0, np.zeros(n * n)
    else:
        dt_old = times[step - 1] - times[step - 2]
        total = dt + dt_old
        w0 = 1 + dt / total
        w1 = -total / dt_old
        w2 = dt**2 / (dt_old * total)
        u_old = history[-2]
    matrix = sp.diags(w0 * psi_flat[active] / dt) + L_active
    rhs = -psi_flat[active] * (w1 * history[-1][active] + w2 * u_old[active]) / dt
    u = history[-1].copy()
    u[active] = spla.spsolve(matrix.tocsc(), rhs)
    history.append(u)

with open("gold/sbm_inactive.csv", "w") as f:
    f.write("time,energy_psi,held_change,mass,mass_psi,max_u,min_u\n")
    for t, u in zip(times, history):
        row = (
            [
                h**2 * (psi_flat * u**2).sum(),
                0.0,
                h**2 * u.sum(),
                h**2 * (psi_flat * u).sum(),
                u.max(),
                u.min(),
            ]
            if t > 0
            else [0.0] * 6
        )
        f.write(f"{t:.10g}," + ",".join(f"{v:.15e}" for v in row) + "\n")
