Coarsened Grain Tracking Design
===============================

Objective
---------
Reduce memory/communication for grain tracking by running labeling/tracking on a coarsened domain (linear factor r, e.g., 4), while retaining fine-grid fields for the PDE. Use a single communication step after local coarse labeling to stitch overlapping labels across ranks.

Scope
-----
- Coarse tracking drives persistence, adjacency, and recoloring decisions.
- Fine grid remains the source for PDE updates; recolor is applied to fine eta.
- Minimal surface area of changes: reuse existing remap pipeline (labeling, stitching, recoloring, remap kernel) with coarse inputs and an upsampled grain_id map.

Data Layout
-----------
- Fine eta: spatial-leading, color-trailing (2D: [ny, nx, n_op], 3D: [nz, ny, nx, n_op]).
- Coarse grid: spatial dims divided by r (assume divisible for now; pad/ceil can be added).
- Coarse ghost width: gh_c = ceil(halo_width / r).

Coarse Aggregation
------------------
- Compute a coarse representation per color:
  - Option A (preferred): block-reduce eta by max over each r³ block per color, then compute masks on coarse eta (max over colors, threshold).
  - Option B: compute fine masks, then block-reduce masks by max/any over r³.
- Tie-breaking in a coarse cell: pick the color with max value; if multiple equal, lowest color index. Background only if all values below threshold.
- Optional: keep a “confidence” flag for mixed blocks (multiple colors above threshold) to allow special handling.

Coarse Labeling
---------------
- Run labelConnectedComponents on coarse masks with connectivity (4/8 or 6/26) and ghost width gh_c.
- Compress labels to contiguous per-color IDs as in the fine pipeline.

Local Metadata (Coarse)
-----------------------
- Compute volume, bbox, centroid in coarse index space, then scale to fine units when needed:
  - bbox_fine_min = bbox_coarse_min * r
  - bbox_fine_max = (bbox_coarse_max + 1) * r - 1
  - centroid_fine = centroid_coarse * r
- Halo bbox: expand coarse bbox by gh_c, clamp to coarse domain; scale to fine if used for adjacency.

Communication (Single Step)
---------------------------
- After local coarse labeling with ghosts, exchange coarse label fields across coarse ghost layers.
- In overlapping ghost regions, record equivalences (rank, coarse_label) pairs for nonzero labels.
- No iterative communication; a single exchange is sufficient because labels are converged locally with ghosts.

Stitching and Persistence
-------------------------
- Use existing union-find stitching on coarse components and equivalence pairs to produce global coarse grain IDs.
- Merge coarse metadata (volume, bbox, centroid) into GrainMeta; store scaled fine-space bbox/centroid for consistency with downstream logic.
- Persistence tracking: centroid-based matching between previous and current GrainMeta (fine-space centroids).

Adjacency and Recoloring
------------------------
- Build adjacency on coarse grains using fine-space bboxes expanded by halo_width (scaled from coarse).
- Greedy recoloring remains unchanged; initial colors come from persistence or defaults.

Mapping Back to Fine Grid
-------------------------
- Build `old_color[grain_id]` / `new_color[grain_id]` from coarse GrainMeta.
- Construct fine grain_id_local:
  - Option A: upsample coarse grain IDs by block repeat (each coarse ID fills its r³ block).
  - Option B: regenerate fine grain_id_local only where coarse IDs change (optional optimization).
- Apply existing remapOrderParameters kernel on fine eta using fine grain_id_local and coarse-derived color maps. No kernel changes needed.

Benefits
--------
- Memory: coarse labels and masks are reduced by ~r³ (e.g., r=4 → 64× less in 3D).
- Communication: single ghost exchange on coarse labels; message sizes shrink by ~r³ and halo shrinks to gh_c.
- Tracking scalability: enables storing full grain ID maps per color on coarse grid.

Risks / Accuracy
----------------
- Small grains (< r in any dimension) may be lost or merged in coarse aggregation; choose r conservatively or flag low-confidence blocks.
- Mixed blocks near interfaces may cause adjacency to miss fine-scale contacts; mitigation options:
  - Lower r (e.g., 2) in sensitive runs.
  - Keep a list of ambiguous coarse blocks and run fine-grain adjacency for them before recoloring.

Integration Plan
----------------
1) Add coarse aggregation helpers:
   - blockMaxDownsample(eta, r) per color
   - computeCoarseMasks(eta, r, threshold)
2) Support coarse ghost width gh_c in labelConnectedComponents (already parameterized by options).
3) Expose a coarse remap entry point:
   - runCoarseRemapStep(eta, options, r, ghost_cb, previous_grains, equivalences)
   - Returns coarse GrainMeta and fine grain_id_local (upsampled).
4) Reuse existing stitching, persistence, adjacency, recoloring, and remap kernels.
5) Add tests:
   - 2D/3D small domains with a diagonal adjacency to validate 8/26 connectivity on coarse grid.
   - Cross-rank slab with a grain crossing the boundary; verify single-step stitching on coarse labels.
   - Coarse vs. fine comparison to quantify small-grain loss and color conflicts.

Communication Impact
--------------------
- Only one exchange: coarse labels with ghost layers (gh_c).
- Equivalence detection uses overlapping coarse ghost cells.
- No per-iteration exchanges; no fine-grid communication added by tracking.

Configuration Knobs
-------------------
- r (coarsening factor), default 4 (2 for sensitive cases).
- connectivity, halo_width.
- aggregation mode (max-based with tie-breaks; optional confidence flag).
- optional mixed-block handling (fine adjacency fallback). 
