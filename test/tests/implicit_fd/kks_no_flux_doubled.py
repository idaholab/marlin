"""Gold of kks_no_flux.i: the mirrored initial condition on the doubled periodic domain.

The strip -20 < x < 20 with zero-flux faces is reproduced by the domain -20 < x < 60, periodic in
x and y, with the initial condition mirrored about x = 20: x is replaced by min(x, 40 - x), the
mirror of x in [20, 60] into [-20, 20]. The solution stays mirror-symmetric, so the integrals over
the doubled domain are twice those of the strip. This script runs kks_no_flux.i on the doubled
domain (no boundary conditions) and writes the halved integrals as the gold; it reproduces the
committed gold to all printed digits.

Usage: python kks_no_flux_doubled.py path/to/marlin-opt   (writes gold/kks_no_flux.csv)
"""

import csv
import os
import subprocess
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
executable = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))

# the strip input's initial conditions with x mirrored about x = 20
x_mirror = "min(x,40-x)"
eta_ic = f"0.5*(1-tanh(2*(sqrt({x_mirror}^2+y^2)-${{r}})/${{l}}))"
# eta_IC is the input's variable, used by the initial conditions of both eta and c
cli_args = [
    "nx=80",
    "Domain/xmax=60",
    "Domain/periodic_directions=X Y",
    "TensorComputes/BCs/active=",
    f"eta_IC={eta_ic}",
    "Outputs/file_base=kks_no_flux_doubled",
]
subprocess.run(
    [executable, "-i", "kks_no_flux.i", "--compute-device=cpu"] + cli_args,
    cwd=here,
    check=True,
)

with open(os.path.join(here, "kks_no_flux_doubled.csv")) as f:
    rows = list(csv.DictReader(f))
with open(os.path.join(here, "gold", "kks_no_flux.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rows[0].keys())
    for row in rows:
        # total_c_change is round-off (the test zeroes it below 1e-10): kept as the doubled run's
        writer.writerow(
            [
                (
                    value
                    if key in ("time", "total_c_change")
                    else f"{float(value) / 2:.14g}"
                )
                for key, value in row.items()
            ]
        )
