"""Gold of bdf2_mode.i: the variable-step BDF2 amplitude of the periodic mode sin(2 pi x)
sin(2 pi y) under u_t = D Laplacian(u), on the n x n cell-centred grid of the unit square.

The mode is an eigenvector of the 3- or 5-point Laplacian with eigenvalue -lambda_h, so its
amplitude follows the scalar recurrence

    (w0 a^{n+1} + w1 a^n + w2 a^{n-1}) / dt + D lambda_h a^{n+1} = 0,

with MOOSE's variable-step BDF2 weights (sum = dt + dt_old)

    w0 = 1 + dt / sum,  w1 = -sum / dt_old,  w2 = dt^2 / (dt_old sum),

and backward Euler (w0 = 1, w1 = -1, w2 = 0) on the first step. The CSV column max_u is the
amplitude times the largest grid value of the mode (0 at t = 0, where the postprocessor has
not run).

Usage: python bdf2_mode.py [stencil]   (writes gold/bdf2_mode_<stencil>pt.csv)
"""

import sys

import numpy as np

D = 0.5
n = 32
stencil = int(sys.argv[1]) if len(sys.argv) > 1 else 3
times = [0.0, 0.01, 0.02, 0.04, 0.05, 0.065, 0.0725, 0.0875, 0.1175]

h = 1.0 / n
k = 2 * np.pi
if stencil == 3:
    per_axis = (2 - 2 * np.cos(k * h)) / h**2
else:
    per_axis = (5 / 2 - 8 / 3 * np.cos(k * h) + 1 / 6 * np.cos(2 * k * h)) / h**2
lam = 2 * per_axis

amplitudes = [1.0]
for step in range(1, len(times)):
    dt = times[step] - times[step - 1]
    if step == 1:
        w0, w1, w2, a_old = 1.0, -1.0, 0.0, 0.0
    else:
        dt_old = times[step - 1] - times[step - 2]
        total = dt + dt_old
        w0 = 1 + dt / total
        w1 = -total / dt_old
        w2 = dt**2 / (dt_old * total)
        a_old = amplitudes[-2]
    amplitudes.append(-(w1 * amplitudes[-1] + w2 * a_old) / (w0 + dt * D * lam))

centres = (np.arange(n) + 0.5) * h
mode_max = np.max(np.sin(k * centres)) ** 2

with open(f"gold/bdf2_mode_{stencil}pt.csv", "w") as f:
    f.write("time,max_u\n")
    for t, a in zip(times, amplitudes):
        f.write(f"{t:.10g},{a * mode_max if t > 0 else 0.0:.15e}\n")
