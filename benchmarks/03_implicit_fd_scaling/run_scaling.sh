#!/usr/bin/env bash
#
# Single-device size scaling of the implicit finite-difference Cahn-Hilliard solve
# (benchmarks/01_spinodal_decomposition/1a_implicit_fd.i) for the PETSC and
# TORCH_PCG linear solvers.
#
# The domain stays [0,200]^2 and the grid is refined (h = 200/n), so each size
# is a different discrete problem from PFHub 1a; the runs are for timing only.
# The biharmonic grows like 1/h^4, so the systems become less diagonally
# dominated with n and Jacobi PCG needs more iterations per solve.
#
# Usage (from this directory, after building marlin-opt):
#   ./run_scaling.sh
#   SIZES="1024 2048" BACKENDS=PETSC ./run_scaling.sh
#   DEVICE=cpu ./run_scaling.sh    # PETSc and torch on a Mac CPU
#
# Results: results/<backend>_<n>.{log,csv} and results/summary.csv. The time per
# step excludes the first step, which carries the startup and solver setup.

set -euo pipefail

# ---- parameters -------------------------------------------------------------
SIZES=${SIZES:-"512 1024 2048 4096 8192"}
# cuda if an NVIDIA GPU is present, mps on macOS, cpu otherwise
if [ -z "${DEVICE:-}" ]; then
  if command -v nvidia-smi > /dev/null 2>&1; then
    DEVICE=cuda
  elif [ "$(uname)" = Darwin ]; then
    DEVICE=mps
  else
    DEVICE=cpu
  fi
fi
# time steps per run (each is `substeps` = 1000 linear solves, as in the input);
# at least 2, since the first step is excluded from the time per step
STEPS=${STEPS:-3}
MARLIN=${MARLIN:-../../marlin-opt}
INPUT=../01_spinodal_decomposition/1a_implicit_fd.i
OUT=results
# -----------------------------------------------------------------------------

# MPS is single precision and has no PETSc: torch PCG only, with the l_tol of the
# MPS tests (single precision cannot reach the input's 1e-8)
extra=()
if [ "$DEVICE" = mps ]; then
  BACKENDS=${BACKENDS:-"TORCH_PCG"}
  extra=(TensorSolver/l_tol=1e-5)
else
  BACKENDS=${BACKENDS:-"PETSC TORCH_PCG"}
fi

mkdir -p "$OUT"
summary="$OUT/summary.csv"
echo "backend,n,dofs,steps,status,wall_time_s,time_per_step_s" > "$summary"

# provenance
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "device: $DEVICE"
  echo "extra arguments: ${extra[*]+"${extra[*]}"}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "$OUT/provenance.txt"

for n in $SIZES; do
  for backend in $BACKENDS; do
    base="$OUT/${backend}_${n}"
    echo "== $backend n=$n ($((n * n)) dofs), $STEPS steps"
    rm -f "$base.csv" "$base.log"

    status=OK
    # ${a[@]+"${a[@]}"}: an empty array expands to nothing under set -u in bash 3.2 (macOS)
    "$MARLIN" -i "$INPUT" \
      Domain/nx=$n Domain/ny=$n \
      Executioner/num_steps=$STEPS \
      TensorSolver/linear_solver=$backend \
      "TensorOutputs/active=''" \
      Outputs/file_base="$base" \
      ${extra[@]+"${extra[@]}"} \
      --compute-device="$DEVICE" > "$base.log" 2>&1 || status="FAILED($?)"

    # wall_time column of the postprocessor CSV (one row per step): the last value and
    # the time per step after the first step
    read -r rows wall per_step < <(
      awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "wall_time") c = i; next }
               c { w[++r] = $c }
               END { printf "%d %s %s\n", r, (r ? w[r] : "-"),
                            (r > 1 ? (w[r] - w[1]) / (r - 1) : "-") }' \
        "$base.csv" 2> /dev/null || echo "0 - -")
    echo "$backend,$n,$((n * n)),$rows,$status,$wall,$per_step" >> "$summary"
    echo "   $status, wall_time=$wall s over $rows steps, $per_step s per step"
  done
done

echo
column -s, -t "$summary"
