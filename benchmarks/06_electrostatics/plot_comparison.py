"""Compare the 6a runs of this input with the PFHub submissions.

usage: python plot_comparison.py <run_dir>

<run_dir> holds the CSV output of the runs listed in OURS (file bases n100, n200, n100_dt05).
The free-energy curves of the PFHub submissions are downloaded into <run_dir>/pfhub_data on the
first call. Writes cost_6a.png (memory against wall time per unit simulated time) and
free_energy_6a.png into the current directory.
"""

import os
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# PFHub 6a submissions (the meta.yaml of each upload, pfhub repository): free-energy data, column
# names, simulated time, wall time (s), memory (MB) and cores as reported by the submitters.
PFHUB = [
    {
        "name": "INL MOOSE, 160^2",
        "url": "https://gist.githubusercontent.com/laagesen/d8ed181db58d77625336461c21f989b3/raw/"
        "9e5679ea74730616d7ee39fb7900a00e1bcc02a6/problem2a_out_160_dtIA8_upload.csv",
        "columns": ("time", "free_energy"),
        "sim_time": 400,
        "wall_time": 495,
        "memory": 800,
        "cores": 8,
    },
    {
        "name": "CORAL (MOOSE)",
        "url": "https://dl.dropbox.com/s/qusutz8g24hx54q/free_energy.csv",
        "columns": ("time", "TotalEnergy"),
        "sim_time": 400,
        "wall_time": 1800,
        "memory": 620,
        "cores": 6,
    },
    {
        "name": "PRISMS-PF",
        "url": "https://gist.githubusercontent.com/wd15/23c6f2d74c4c41d8da883feee30efdee/raw/"
        "12633176adddfb4375c780cfa10678d251520d3d/energy_6a.csv",
        "columns": ("Time", "Energy"),
        "sim_time": 400,
        "wall_time": 507,
        "memory": 124,
        "cores": 16,
    },
    {
        "name": "MMSP, dx = 1",
        "url": "https://gist.githubusercontent.com/tkphd/0e6057ada4de47098d5d39a27d8866ab/raw/"
        "fc51bc61c389df0b21341781e5f60bd098e09aa5/energy_6a_dx10.csv",
        "columns": ("time", "free_energy"),
        "sim_time": 1000,
        "wall_time": 32640,
        "memory": 2.5,
        "cores": 12,
    },
    {
        "name": "MEMPHIS, 100^2",
        "url": "https://raw.githubusercontent.com/sqmonismith/Benchmark_06/main/free_energy.csv",
        "columns": ("time", "free_energy"),
        "sim_time": 1000,
        "wall_time": 658,
        "memory": 7.6,
        "cores": 1,
    },
    {
        "name": "FiPy, 100^2",
        "url": "https://gist.githubusercontent.com/wd15/db0d5558e0b60aeade766f0f847cfd06/raw/"
        "c0088b2f1cd67e8774b54150eab96b854de13c96/energy.txt",
        "columns": ("time", "free_energy"),
        "sim_time": 1000,
        "wall_time": 40,
        "memory": 164,
        "cores": 1,
    },
]

# Runs of 6a_implicit_fd.i to t = 400 on one core of an Apple M5 Pro (serial PETSc, CPU),
# 2026-10-28, run with the arguments n=100, n=200 and n=100 Executioner/dtmax=0.05. Wall time and
# memory from the "Finished Executing" line of the console output.
OURS = [
    {
        "name": "Marlin, 100^2, dtmax 0.5",
        "file_base": "n100",
        "wall_time": 24.5,
        "memory": 227,
    },
    {
        "name": "Marlin, 200^2, dtmax 0.5",
        "file_base": "n200",
        "wall_time": 77.3,
        "memory": 270,
    },
    {
        "name": "Marlin, 100^2, dtmax 0.05",
        "file_base": "n100_dt05",
        "wall_time": 71.8,
        "memory": 242,
    },
]
OURS_SIM_TIME = 400
OURS_STYLES = ["-", "--", ":"]
COLORS = ["#999999", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]  # Okabe-Ito


def read_curve(path, columns):
    """Time and free energy from a CSV or whitespace-separated table with a header row."""
    with open(path) as fh:
        header = fh.readline()
    delimiter = "," if "," in header else None
    data = np.genfromtxt(path, delimiter=delimiter, names=True)
    t, f = data[columns[0]], data[columns[1]]
    keep = t > 0  # t = 0 is the initial state (CORAL reports 0 there); the time axis is logarithmic
    return t[keep], f[keep]


run_dir = sys.argv[1]
data_dir = os.path.join(run_dir, "pfhub_data")
os.makedirs(data_dir, exist_ok=True)
for ref in PFHUB:
    # several uploads share a file name, so name each copy after its submission
    ref["path"] = os.path.join(data_dir, ref["name"].split(",")[0].split(" ")[0] + ".csv")
    if not os.path.exists(ref["path"]):
        urllib.request.urlretrieve(ref["url"], ref["path"])

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    }
)

# Figure 1: memory against wall time per unit simulated time
LABEL_OFFSETS = {"FiPy, 100^2": (6, -20), "INL MOOSE, 160^2": (6, 12)}
fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
for ref, color in zip(PFHUB, COLORS):
    rate = ref["wall_time"] / ref["sim_time"]
    ax.scatter(rate, ref["memory"], s=45, color=color, edgecolor="k", lw=0.5, zorder=3)
    ax.annotate(
        f"{ref['name']}\n{ref['cores']} core{'s' if ref['cores'] > 1 else ''}, t = {ref['sim_time']}",
        (rate, ref["memory"]),
        xytext=LABEL_OFFSETS.get(ref["name"], (6, 4)),
        textcoords="offset points",
        fontsize=8,
    )
for run, marker in zip(OURS, ["*", "s", "^"]):
    rate = run["wall_time"] / OURS_SIM_TIME
    ax.scatter(
        rate,
        run["memory"],
        s=40,
        marker=marker,
        color="k",
        zorder=4,
        label=run["name"] + ", 1 core",
    )
ax.legend(loc="lower left", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-2, 2e2)
ax.set_ylim(1, 3e3)
ax.set_xlabel("wall time / simulated time (s per unit time)")
ax.set_ylabel("memory (MB)")
ax.set_title("PFHub 6a: cost as reported (hardware from 2017 to 2022, ours 2026)", fontsize=10)
fig.savefig("cost_6a.png", dpi=150)

# Figure 2: free energy against time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax in (ax1, ax2):
    for ref, color in zip(PFHUB, COLORS):
        t, f = read_curve(ref["path"], ref["columns"])
        ax.plot(
            t,
            f,
            color=color,
            lw=1.2,
            marker="o" if len(t) < 200 else None,
            ms=2.5,
            label=ref["name"],
        )
    for run, ls in zip(OURS, OURS_STYLES):
        t, f = read_curve(os.path.join(run_dir, run["file_base"] + ".csv"), ("time", "free_energy"))
        ax.plot(t, f, color="k", lw=1.8, ls=ls, label=run["name"])
ax1.set_xscale("log")
ax1.set_xlim(1e-2, 400)
ax1.set_xlabel("time")
ax1.set_ylabel("free energy F")
ax1.set_title("(a) whole run, t to 400")
ax2.set_xlim(200, 400)
ax2.set_ylim(100, 128)
ax2.set_xlabel("time")
ax2.set_title("(b) late times")
ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.suptitle("PFHub 6a (square domain): free energy against time")
fig.savefig("free_energy_6a.png", dpi=150)
