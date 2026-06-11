# Grain Tracking and Remapping

## Context

Marlin runs fixed-grid phase-field simulations with libtorch tensors (spatial
dimensions leading, in x, y, z order). Polycrystalline models represent grains by
order parameters (OPs); to keep the number of OPs small, multiple grains share an
OP as long as they are spaced sufficiently far apart. The grain tracker

1. detects grains (connected components where one OP is maximal and above a
   threshold),
2. tracks their identity across time steps (persistent ids),
3. detects when two grains sharing an OP approach each other closer than an
   exclusion distance, and
4. moves one of them to a free OP without perturbing the solution.

The design follows MOOSE's `GrainTracker`, recast for dense-grid tensor data with
GPU-resident field operations and host-side graph/bookkeeping operations.

## User interface

The `GrainTracker` compute runs in the `[TensorComputes/Postprocess]` group:

```
[TensorComputes]
  [Postprocess]
    [tracker]
      type = GrainTracker
      op_buffers = 'eta0 eta1 eta2'  # one buffer per order parameter
      threshold = 0.1                # detection threshold
      halo_width = 4                 # exclusion distance in cells
      connectivity = minimal         # face neighbors (or 'full' incl. corners)
      interval = 5                   # run every 5 steps
      grain_id_buffer = grain_id     # optional: per-cell persistent grain id
      on_conflict = error            # or 'warn'
    []
  []
[]
```

The order parameter buffers are remapped **in place**, including old states (time
integrator history, `remap_old_states = true`). Scalar diagnostics are available
through `GrainTrackerPostprocessor` (`value_type = count | remapped | conflicts`).

`halo_width` is the exclusion distance in cells: grains whose footprints, dilated
by `halo_width`, touch are considered conflicting and may not share an OP. The
remap also moves the field within this dilated footprint, so it should span a few
diffuse interface widths.

### Parallelism and periodicity

- **Serial (any parallel mode):** periodicity (from `Domain/periodic_directions`)
  is handled by in-tensor wrap in all stages.
- **REAL_SPACE parallel mode:** grains are stitched across rank boundaries (and
  periodic boundaries) via ghost layer exchange; every rank computes the identical
  global grain list deterministically (no coordinator rank).
- **Spectral parallel modes (FFT_SLAB/FFT_PENCIL):** not supported with more than
  one rank (no ghost communication); the tracker errors out.

## Algorithm

All stages operate on the stacked field `eta` ([*spatial*, n_op]) assembled from
the op buffers, cropped to the owned region plus a `halo_width`-wide ghost ring.

1. **Masks (device):** per OP `c`: `argmax(eta) == c & max(eta) > threshold`.
   Ghost rings along dimensions that no exchange refreshes (un-partitioned,
   non-periodic) are blanked.
2. **Connected component labeling (device):** iterative minimum-label propagation
   (Bellman-Ford style) per OP with optional periodic wrap (`torch::roll`),
   followed by compaction to contiguous ids.
3. **Global numbering:** per-rank per-OP component counts are allgathered;
   prefix sums give globally unique label ids without communication of the grids.
4. **Moments (device):** per-label volume and coordinate sums over *owned* cells
   in global coordinates. For periodic dimensions, circular sums
   (sin/cos of the angle-mapped coordinate) are accumulated; all moments are
   additive and are combined with a single `MPI_Allreduce`. Volume-weighted
   circular means give exact centroids for wrap-spanning grains.
5. **Seam stitching:** the label grid is ghost-exchanged
   (`HaloCommunication::exchangeGhostTensor`, which also wrap-copies periodic
   single-partition dimensions). Cells in the halo where the pre- and
   post-exchange labels differ yield equivalence pairs; pairs are allgathered and
   merged with a union-find into dense grain ids (deterministic: ordered by
   smallest label).
6. **Persistence:** mutual-nearest-centroid matching against the previous step
   with minimum-image distances on periodic dimensions and a relative volume
   guard. Matched grains keep their persistent id; unmatched grains get new ids.
   The OP a grain occupies is always taken from detection (argmax), never from
   tracking bookkeeping.
7. **Adjacency (device):** the grain id grid is expanded by `halo_width` using
   breadth-first label fronts (cells belong to the nearest grain; first-arrival
   ownership, never overwritten), ghost-exchanged, and scanned for neighboring
   distinct ids. Pairs are allgathered into a global adjacency graph.
8. **Recoloring (host):** deterministic greedy coloring that keeps the current OP
   where possible and otherwise assigns the smallest OP not used by any neighbor;
   grains are visited largest-first. Unresolvable conflicts (more mutually close
   grains than OPs) are reported via `on_conflict = error | warn`.
9. **Remap (device):** only grains whose OP changed are touched. Within each such
   grain's expanded (nearest-grain) footprint the old channel value is zeroed and
   max-combined into the new channel. Cells outside changed footprints —
   including sub-threshold interface tails of all other grains — are bit-identical
   untouched. Old states are remapped with the same mapping. When nothing changed
   the field is not written at all.

## Library layer

`GrainRemap` (in `include/utils/GrainRemap.h`) provides the building blocks as
free functions operating on plain tensors plus a `Geometry` descriptor
(owned extents, global extents, global offset, halo padding, periodicity):

- `computeColorMasks`, `labelConnectedComponents`, `buildGlobalContiguousLabels`
- `expandLabels` (breadth-first, portable shift-max implementation — no pooling
  ops, works for int64 on CPU/CUDA/MPS)
- `computeComponentMoments`, `detectSeamEquivalences`, `mergeLabels`,
  `finalizeGrains` (circular-mean centroids)
- `matchPersistentGrains`, `extractAdjacencyPairs`, `buildAdjacencyLists`,
  `greedyRecolor`, `applyLabelMap`, `remapOrderParameters`
- `runRemapStep`: complete serial pipeline (used by unit tests and useful for
  scripting); the distributed pipeline lives in the `GrainTracker` compute.

`ComponentMoments` is `pack()`/`unpack()`-able for MPI reductions.

## Testing

- Unit tests (`unit/src/GrainRemapTest.C`) cover each building block: wrap-aware
  labeling, first-arrival expansion, exact moments (incl. owned-region cropping,
  global offsets, and periodic centroids), seam equivalence detection and
  merging, deterministic recoloring with conflict reporting, non-destructive
  remapping, persistence matching (periodic, volume guard), and end-to-end
  scenarios: close same-OP grains are separated (incl. across periodic
  boundaries and in 3D), far grains share OPs with a bit-identical field.
- Regression tests (`test/tests/grain_tracker/`) exercise the input-file
  interface in 2D, 3D, with periodic boundaries, and in REAL_SPACE parallel runs.

## Future work

- Coarse-grid tracking for large domains (see `grain_remap_coarsening.md`):
  run labeling/tracking on a block-reduced grid and upsample the grain id map.
- Faster CCL via pointer jumping if grain diameters grow large (the current
  propagation needs O(diameter) device passes).
- Halo-overlap based tracking (instead of centroid matching) for very fast moving
  or splitting/merging grains.
