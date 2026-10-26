"""PFHub 6a outputs from a 6a_implicit_fd.i run: the free energy against time, and c and phi on
the cuts x = 50 and y = 50 at the last output time.

Usage: python cut_planes.py <file_base>   (reads <file_base>.csv and <file_base>.h5)

The fields are cell centred on an n x n grid of [0, 100]^2, so x = 50 is a cell face for even n;
the cut takes the mean of the two cells next to it.
"""

import sys

import h5py
import numpy as np

L = 100.0

base = sys.argv[1]

# free energy; the row at t = 0 is that of the initial guess of phi (phi is solved from the first
# step on), so it is not the benchmark's initial energy and is left out
data = np.genfromtxt(base + ".csv", delimiter=",", names=True)
data = data[data["time"] > 0]
np.savetxt(
    base + "_free_energy.csv",
    np.column_stack([data["time"], data["free_energy"]]),
    delimiter=",",
    header="time,free_energy",
    comments="",
)

with h5py.File(base + ".h5", "r") as f:
    last = max(int(name.split(".")[1]) for name in f if name.startswith("c."))
    c = np.array(f[f"c.{last}"])
    phi = np.array(f[f"phi.{last}"])

n = c.shape[0]
h = L / n
centres = (np.arange(n) + 0.5) * h

# axis order of the stored arrays: phi = sin(y / 7) on the plate x = L, so the x axis is the one
# along which the last layer matches it
x_axis = (
    0
    if np.abs(phi[-1, :] - np.sin(centres / 7)).max()
    < np.abs(phi[:, -1] - np.sin(centres / 7)).max()
    else 1
)
if x_axis == 1:
    c, phi = c.T, phi.T


def cut(field, at, axis):
    """the field on the line where coordinate 'axis' equals 'at' (axis 0 is x)"""
    i = at / h - 0.5
    lo = int(np.floor(i))
    w = i - lo
    rows = np.take(field, [lo, lo + 1], axis=axis)
    return (1 - w) * np.take(rows, 0, axis=axis) + w * np.take(rows, 1, axis=axis)


for name, axis, along in (("x_cut_plane", 0, "y"), ("y_cut_plane", 1, "x")):
    np.savetxt(
        f"{base}_{name}.csv",
        np.column_stack([centres, cut(c, 50.0, axis), cut(phi, 50.0, axis)]),
        delimiter=",",
        header=f"{along},concentration,potential",
        comments="",
    )
print(f"t index {last}, n = {n}, x axis {x_axis}")
