Grain Remapping Design and Usage
================================

Context
-------
Marlin runs fixed-grid phase-field simulations using libtorch tensors with spatial dimensions leading and order parameter (color) as the trailing dimension (2D: [ny, nx, n_op], 3D: [nz, ny, nx, n_op]). Grains are connected components where a single color is both maximal and above a threshold. The remapper tracks grains across time and recolors them to prevent conflicts when grains of the same color become neighbors. It is designed for multi-GPU/MPI runs with existing halo exchange.

Core Data Structures
--------------------
- GrainRemapOptions: spatial_dim (2 or 3), n_colors, threshold, halo_width, connectivity (4/8 or 6/26), tracking_tolerance.
- ComponentMeta: per-rank component info (rank, color, local_label, volume, bbox, halo bbox, centroid).
- GrainMeta: global grain info (grain_id, persistent_id, old_color, new_color, volume, bbox, centroid).
- ComponentRef + UnionFind: host-side stitching of cross-rank component equivalences.
- grain_id_local: device int tensor matching spatial shape, stores global grain id per cell (-1 background).

Algorithm Stages
----------------
1) Per-color masks (GPU): compute `max` over trailing color dim; mask_c = (argmax == c) & (eta[..., c] > thresh) & (max_val > thresh).
2) Connected components (GPU): initialize labels with unique id per foreground cell (+1) and 0 for background. Iterate Bellman–Ford style neighbor-min propagation until convergence (connectivity controlled by options). Compress labels to contiguous ids (0..n_local_c-1). MPS fallback uses CPU `_unique`.
3) Local metadata (host): for each local label, accumulate volume, bbox, centroid, halo bbox (bbox expanded by halo_width but clamped to domain). Implemented on CPU for simplicity and determinism.
4) Halo exchange (MPI): use `HaloCommunication::exchangeGhostTensor` to exchange label fields across ghost layers; caller supplies the callback so existing ghost exchange infrastructure is reused.
5) Cross-rank stitching: caller provides equivalence pairs between touching components across ranks; UnionFind merges them into global grain ids. Merge component metadata into GrainMeta (volume sum, bbox union, volume-weighted centroid).
6) Persistence tracking: compare current grains vs. previous GrainMeta list on a coordinator (e.g., rank 0) by centroid distance; if mutual nearest within tolerance, keep persistent_id and color; otherwise assign new persistent_id and default color.
7) Adjacency and recoloring: build adjacency between grains of the same color using expanded bbox overlap (halo_width). Greedy recoloring prefers keeping current color; assigns smallest free color not used by neighbors (up to n_colors). For debug/visualization a dense adjacency can be built directly from halo-expanded label grids with `buildHaloAdjacency`; this feeds PETSc coloring via `colorAdjacencyWithPetsc`.
8) Remap fields (GPU): build device tensors old_color[new_color] indexed by grain id. Flatten spatial dims, launch remap kernel: for each cell idx, read grain_id_local[idx], map old->new color, copy eta_old(idx, old_color) into eta(idx, new_color). Background stays zero; no write conflicts (one-to-one mapping).

Key Functions (namespace GrainRemap)
------------------------------------
- computeColorMasks(eta, threshold) -> vector<torch::Tensor>
- labelConnectedComponents(mask, options) -> torch::Tensor (int64, -1 background)
- labelConnectedComponentsWithRaw(mask, options) -> pair(raw_labels>=0, compact_labels=-1 background)
- buildGlobalContiguousLabels(per_color_labels, offsets) -> combined labels (global unique, -1 background). Per-color labels must already be compact 0..N-1; offsets[color] gives the global id base.
- combineRawLabelsAcrossColors(raw_labels) -> combined raw ids (background 0) if you need the pre-compaction view.
- dilateMask(mask, halo_width) -> boolean dilation (no wrap); expandLabelsWithHalo(labels, halo_width) -> integer dilation preserving label ids in the halo.
- buildHaloAdjacency(halo_labels, connectivity) -> dense CPU adjacency and unique label list from a halo-expanded label grid (background -1); useful for dumps or feeding PETSc coloring.
- buildOldColorTable(per_color_labels, offsets, n_colors) / buildNewColorTable(unique_labels, colors) -> 1D CPU tensors mapping label id -> color; buildLabelColorGrid(labels, unique_labels, colors) -> grid of per-cell colors for visualization.
- computeComponentMetadata(labels, color, halo_width, rank) -> vector<ComponentMeta>
- mergeComponents(components, equivalences, n_colors, component_to_grain)
- labelsToGlobalIds(labels, label_to_global, options) -> torch::Tensor (int32)
- matchPersistentGrains(previous, current, tolerance) -> vector<persistent_id>
- buildAdjacency(grains, halo_width) -> adjacency list
- greedyRecolor(adjacency, initial_colors, n_colors)
- remapOrderParameters(eta, grain_ids, old_colors, new_colors)
- colorAdjacencyWithPetsc(adjacency, n_colors, algorithm) -> vector<unsigned int> using PETSc MatColoring on a CPU adjacency tensor (avoids building PolycrystalICTools::AdjacencyMatrix).
- runRemapStep(eta, options, rank, ghost_exchange_cb, previous_grains, equivalences, chosen_colors)

Halo Exchange Utility
---------------------
`HaloCommunication::exchangeGhostTensor` mirrors TensorProblem ghost logic: validates halo sizes, handles periodic wrap for single-partition dims, GPU-aware MPI when available, symmetric halo width inferred from tensor shape vs. owned size. Works for arbitrary tensors (e.g., labels).

Torch Operations Used (selection)
---------------------------------
- Masks: `eta.max(-1)` for trailing color, `argmax == c`, thresholding with `>` and `&`.
- Label init: `torch::arange(numel).view(mask.shape) + 1`, `torch::where`.
- Propagation: `torch::where` over sliced neighbor views; `torch::any(updated != labels)` to detect convergence.
- Compaction: `torch::_unique` (CPU fallback on MPS), host-built map applied with `map.index({labels})`.
- Global merge: `torch::where` with per-color offsets in `buildGlobalContiguousLabels`.
- Halo expansion: `max_pool2d/3d` with stride 1/padding 1 in `expandLabelsWithHalo`.
- Remap: flatten spatial dims; per-cell kernel copies `eta_old(idx, old_color)` -> `eta(idx, new_color)`.

Integrating a Remap Step
------------------------
Typical call on each remap event:
1) Prepare GrainRemapOptions (spatial_dim, n_colors, threshold, halo_width, connectivity, tracking_tolerance).
2) Provide a ghost exchange callback: `auto ghost_cb = [&](torch::Tensor & t, unsigned int gh){ HaloCommunication::exchangeGhostTensor(t, gh, domain); };`
3) Detect inter-rank component equivalences after exchanging labels (ghost overlap where both labels nonzero).
4) Call `runRemapStep(eta, options, rank, ghost_cb, previous_grains, equivalences, &colors_out);`
5) Keep `result.grain_ids` for PDE steps (neighbor detection, postprocessing).
6) On coordinator, store `result.grains` as current metadata for next remap (persistence).

Dimensional and Layout Assumptions
----------------------------------
- Spatial dims leading, color trailing; flatten spatial dims only, never permute color dim.
- 2D shape [ny, nx, n_op], 3D shape [nz, ny, nx, n_op]; dtype float32.
- Labels and grain_id tensors mirror spatial dims only.
- Connectivity: 4/8 in 2D, 6/26 in 3D; halos sized by halo_width.

Performance Notes
-----------------
- GPU vs CPU:
  - GPU: per-color masks, connected-component propagation (iterative neighbor minima), halo expansion (`expandLabelsWithHalo` via max-pool), device-side adjacency pair extraction (`buildHaloAdjacency` when the halo labels are on device), and the final remap kernel. Complexity: masks O(Ncells), propagation O(Ncells * Niter * nneigh) with small neighbor count (4/8/6/26) and typically few iterations, halo expansion O(Ncells * halo_width), remap O(Ncells).
  - CPU: unique-label compression (always; MPS forces `_unique` CPU fallback), host metadata (volume/bbox/centroid), union-find stitching, persistence matching, greedy recolor, and dense adjacency assembly after the device has produced neighbor pairs. Complexity: metadata O(Ncells_local) on CPU; union-find O(Ncomp α(Ncomp)); greedy recolor O(E * Npasses) with small E from bbox/halo coarse tests; PETSc coloring cost depends on chosen algorithm.
  - Memory: halo-expanded labels are reused to avoid extra copies; device adjacency build only transfers a compact list of neighboring label pairs to host.
- No atomics are needed in remap kernel (one source->one destination).
- Greedy coloring is O(E) per pass; adjacency is built from bounding boxes or halo overlap, not full per-cell scans on CPU. The optional dense adjacency from halo grids is only used for debugging/PETSc coloring and is constructed from device-generated pairs.

PETSc-based Coloring (optional)
-------------------------------
- If PETSc is available, `colorAdjacencyWithPetsc` can color the grain graph using `Moose::PetscSupport::colorAdjacencyMatrix`, which wraps `MatColoring`. Pass a CPU adjacency tensor (square) and the desired coloring algorithm (e.g., `"power"`, `"jp"`, `"greedy"`). The helper zeroes the diagonal and copies the adjacency into a dense PETSc matrix without creating an intermediate `AdjacencyMatrix` object.

Parallel Stitching Notes
------------------------
- Do per-rank labeling and compaction per color, then call `buildGlobalContiguousLabels` to make labels unique across colors on that rank (background -1).
- Exchange these labels across ghost layers (with one ghost cell is usually enough for adjacency).
- Detect inter-rank equivalences at ghost overlaps: if both local and ghost labels are >=0 and differ, record (rankA,labelA,colorA) ~ (rankB,labelB,colorB). Use UnionFind to merge and assign final global grain ids, avoiding duplicate ids.
- Coloring can be done locally before stitching to reduce conflicts quickly; final recolor should use stitched grain ids if cross-rank adjacency matters.
- When renumbering after stitching, reuse the per-color offsets plus the union-find mapping so that each stitched grain gets a single global id and its field values can be remapped consistently.

Extensibility / Hooks
---------------------
- Equivalence detection is caller-provided after ghost exchange; adapt to your MPI layout.
- Persistence matching can be extended for periodicity or more advanced heuristics (e.g., overlap).
- Metadata computation can be replaced by GPU reductions if needed; current CPU path favors simplicity and determinism.

Testing
-------
- Unit tests (`unit/src/GrainRemapTest.C`) cover masks, connectivity labeling (4 vs. 8), and remap correctness using MooseTensor device options to run on CPU/CUDA/MPS as available.
- Larger integration-style tests (`unit/src/GrainRemapLargeLabelTest.C`) build synthetic multi-disk/blob fields, run per-color labeling/compaction, global contiguity, halo expansion, dense adjacency construction with `buildHaloAdjacency`, PETSc coloring, and the order-parameter remap. Optional dumps (`GRAIN_LABEL_DUMP`, `GRAIN_LABEL_DUMP_2D`) save intermediate tensors for inspection.

Common Questions
----------------
- Q: Why CPU in some steps? A: Metadata and MPS `_unique` fallback favor correctness and portability; hot loops (propagation, remap) stay on device.
- Q: How to avoid color conflicts? A: The greedy recolor uses adjacency of same-color grains; initial colors are preserved when free.
- Q: Do I need new build deps? A: No. Uses existing libtorch, CUDA/MPS/CPU backends, MPI layer via `HaloCommunication`.
