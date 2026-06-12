/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace GrainRemap
{

/**
 * Geometry of the local tensor slab within the global domain. All extents are in
 * cells and given in tensor dimension order (i.e. [d0, d1(, d2)] matching the
 * leading spatial dimensions of the field tensors).
 *
 * Local tensors may carry a symmetric halo padding of `pad` cells on every spatial
 * dimension (Marlin REAL_SPACE mode); the owned region is
 * tensor.narrow(d, pad, owned[d]) per dimension.
 */
struct Geometry
{
  int spatial_dim = 2;
  /// owned (un-padded) local extent per spatial dimension
  std::array<int64_t, 3> owned{{0, 0, 0}};
  /// global domain extent per spatial dimension
  std::array<int64_t, 3> global{{0, 0, 0}};
  /// global index of the first owned cell per spatial dimension
  std::array<int64_t, 3> global_begin{{0, 0, 0}};
  /// allocated symmetric halo padding of the local tensors
  int64_t pad = 0;
  /// physical periodicity of the global domain per spatial dimension
  std::array<bool, 3> periodic{{false, false, false}};

  /// geometry for a serial, un-padded tensor covering the whole domain
  static Geometry serial(const torch::Tensor & spatial_field,
                         const std::array<bool, 3> & periodic = {{false, false, false}});
};

/// Options controlling grain detection and remapping.
struct GrainRemapOptions
{
  /// phase-field threshold for grain detection
  double threshold = 0.1;
  /// neighbor connectivity: 4/8 in 2D, 6/26 in 3D (0 = automatic face connectivity)
  int connectivity = 0;
  /// exclusion / remap halo distance in cells: grains closer than this (after
  /// dilation by halo_width) are considered neighbors and must not share an
  /// order parameter; remap moves values within the grain dilated by halo_width
  int halo_width = 2;
  /// max centroid distance (cells) for matching grains between remap steps
  double tracking_tolerance = 3.0;
  /// max relative volume change for accepting a persistence match (0 disables the check)
  double tracking_volume_ratio = 4.0;
  /// in-tensor periodic wrap per spatial dimension. Use for dimensions where the
  /// local tensor spans the full periodic domain *and* no ghost-based stitching
  /// is performed (serial / un-partitioned dimensions without halo exchange).
  std::array<bool, 3> wrap{{false, false, false}};

  /// default connectivity (face neighbors) for a given spatial dimension
  static int defaultConnectivity(int spatial_dim) { return spatial_dim == 2 ? 4 : 6; }
};

/**
 * Additive per-component moments. Sums are accumulated in global cell coordinates;
 * for periodic dimensions the circular sums (sin/cos of the angle-mapped coordinate)
 * allow exact volume-weighted centroids of wrap-spanning grains. All fields are
 * additive under component merging and MPI reduction.
 */
struct ComponentMoments
{
  double volume = 0.0;
  std::array<double, 3> sum{{0.0, 0.0, 0.0}};
  std::array<double, 3> sum_sin{{0.0, 0.0, 0.0}};
  std::array<double, 3> sum_cos{{0.0, 0.0, 0.0}};

  /// number of doubles in the packed representation
  static constexpr int packed_size = 10;
  void pack(double * out) const;
  void unpack(const double * in);
  ComponentMoments & operator+=(const ComponentMoments & other);
};

/// Aggregated (global) grain metadata.
struct GrainMeta
{
  int64_t grain_id = -1;      ///< grain id for this remap step (dense 0..N-1)
  int64_t persistent_id = -1; ///< stable id across remap steps
  int color = -1;             ///< order parameter the grain currently occupies (detected)
  int new_color = -1;         ///< order parameter after recoloring
  int64_t volume = 0;         ///< volume in cells
  std::array<double, 3> centroid{{0.0, 0.0, 0.0}}; ///< global cell coordinates
};

/// Standard disjoint set with path compression and union by rank.
class UnionFind
{
public:
  explicit UnionFind(std::size_t n);
  std::size_t find(std::size_t i);
  void unite(std::size_t a, std::size_t b);
  std::size_t size() const { return _parent.size(); }

private:
  std::vector<std::size_t> _parent;
  std::vector<std::size_t> _rank;
};

/// Per-color mask: true where eta[..., c] is maximal and above the threshold.
std::vector<torch::Tensor> computeColorMasks(const torch::Tensor & eta, double threshold);

/**
 * Label connected components of a boolean mask with iterative minimum-label
 * propagation. Dimensions flagged in `wrap` are treated as periodic (the first
 * and last layer are neighbors). Returns compact labels 0..N-1 with background -1.
 */
torch::Tensor labelConnectedComponents(const torch::Tensor & mask,
                                       int connectivity,
                                       const std::array<bool, 3> & wrap = {
                                           {false, false, false}});

/**
 * Combine per-color compact labels (background -1, ids 0..Nc-1) into a single grid
 * with contiguous ids across colors (background -1). offsets[c] gives the id base
 * of color c; counts[c] the number of components of color c.
 */
torch::Tensor buildGlobalContiguousLabels(const std::vector<torch::Tensor> & per_color_labels,
                                          std::vector<int64_t> & offsets,
                                          std::vector<int64_t> & counts);

/**
 * Expand non-negative integer labels outward by `steps` cells (Chebyshev distance)
 * using breadth-first label fronts: cells are claimed by the first front that
 * reaches them (ties towards the larger id) and are never overwritten, so the
 * expanded regions form a nearest-grain partition. Background must be -1.
 * Portable to CPU/CUDA/MPS (no pooling ops). Dimensions flagged in `wrap` expand
 * periodically.
 */
torch::Tensor expandLabels(const torch::Tensor & labels,
                           int steps,
                           const std::array<bool, 3> & wrap = {{false, false, false}});

/**
 * Compute additive moments for each label id in [0, n_labels). Only cells inside
 * the owned region (geometry pad / owned extents) are counted; coordinates are
 * global (local index - pad + global_begin). Labels must be a spatial integer
 * tensor with background -1.
 */
std::vector<ComponentMoments> computeComponentMoments(const torch::Tensor & labels,
                                                      int64_t n_labels,
                                                      const Geometry & geom);

/**
 * Detect label equivalences along the halo seams after a ghost exchange.
 * `pre` is the local label grid before the exchange (labels extend into the halo
 * region by local labeling), `post` the grid after the exchange (halo cells hold
 * the neighbor's labels for the same physical cells). Both must hold globally
 * unique ids with background -1. Returns unique (a, b) pairs with a < b.
 */
std::vector<std::pair<int64_t, int64_t>> detectSeamEquivalences(const torch::Tensor & pre,
                                                                const torch::Tensor & post,
                                                                const Geometry & geom);

/**
 * Merge labels into dense grain ids using the given equivalence pairs.
 * Returns the label -> grain id map (size n_labels) and sets n_grains.
 * Grain ids are assigned in order of the smallest label in each set, making the
 * result deterministic and identical on all ranks for identical inputs.
 */
std::vector<int64_t> mergeLabels(int64_t n_labels,
                                 const std::vector<std::pair<int64_t, int64_t>> & equivalences,
                                 int64_t & n_grains);

/**
 * Aggregate per-label moments into per-grain metadata. label_color[l] gives the
 * order parameter each label was detected on; merged labels must share a color.
 * Centroids are finalized from the (possibly circular) moment sums.
 */
std::vector<GrainMeta> finalizeGrains(const std::vector<ComponentMoments> & label_moments,
                                      const std::vector<int64_t> & label_to_grain,
                                      const std::vector<int> & label_color,
                                      int64_t n_grains,
                                      const Geometry & geom);

/**
 * Track grains across remap steps by mutual-nearest centroid matching with
 * minimum-image distances for periodic dimensions and a relative volume guard.
 * Sets persistent_id on `current`; colors are not modified (the detected color is
 * authoritative). Unmatched grains receive fresh ids starting at
 * max(first_new_id, max persistent id in `previous` + 1); pass a monotone
 * counter as first_new_id to prevent ids of vanished grains from being reused.
 * Returns the persistent ids.
 */
std::vector<int64_t> matchPersistentGrains(const std::vector<GrainMeta> & previous,
                                           std::vector<GrainMeta> & current,
                                           const GrainRemapOptions & options,
                                           const Geometry & geom,
                                           int64_t first_new_id = 0);

/**
 * Extract unique adjacency pairs (grain_a, grain_b), a < b, from a grid of
 * (typically halo-expanded) grain ids with background -1.
 */
std::vector<std::pair<int64_t, int64_t>> extractAdjacencyPairs(const torch::Tensor & grain_ids,
                                                               int connectivity,
                                                               const std::array<bool, 3> & wrap = {
                                                                   {false, false, false}});

/// Build adjacency lists from unique pairs.
std::vector<std::vector<int64_t>> buildAdjacencyLists(
    int64_t n_grains, const std::vector<std::pair<int64_t, int64_t>> & pairs);

/**
 * Apply a host-side label map to a label grid: cells with label l >= 0 become
 * map[l], background (-1) is preserved. Out-of-range labels are an error in the
 * caller's bookkeeping; labels are clamped to 0 for the gather, so the map must
 * cover [0, max_label].
 */
torch::Tensor applyLabelMap(const torch::Tensor & labels, const std::vector<int64_t> & map);

/**
 * Greedy graph recoloring that keeps the initial color where possible and otherwise
 * assigns the smallest color in [0, n_colors) not used by any neighbor. Vertices
 * are visited in the given order (pass e.g. descending volume). Deterministic.
 * Grains for which no conflict-free color exists keep their color and are appended
 * to `conflicts`.
 */
std::vector<int> greedyRecolor(const std::vector<std::vector<int64_t>> & adjacency,
                               const std::vector<int> & initial_colors,
                               int n_colors,
                               const std::vector<int64_t> & order,
                               std::vector<int64_t> & conflicts,
                               unsigned int max_passes = 8);

/**
 * Move grains to their new order parameters. Only cells covered by
 * `changed_grain_ids` (grain ids of recolored grains, typically the grain id grid
 * masked to changed grains and dilated by halo_width; background -1) are touched:
 * the value of the grain's old channel is zeroed and combined into the new channel
 * via max. Cells outside changed regions, including diffuse interface tails of
 * unchanged grains, are left untouched.
 */
void remapOrderParameters(torch::Tensor & eta,
                          const torch::Tensor & changed_grain_ids,
                          const std::vector<int> & old_colors,
                          const std::vector<int> & new_colors);

/// Result of a complete remap step.
struct RemapResult
{
  /// device tensor with the spatial shape of eta, int64, -1 background
  torch::Tensor grain_ids;
  /// aggregated grain metadata (with final colors)
  std::vector<GrainMeta> grains;
  /// grains whose order parameter changed in this step
  std::vector<int64_t> remapped_grains;
  /// grains that could not be assigned a conflict-free order parameter
  std::vector<int64_t> conflicts;
};

/**
 * Convenience driver for the complete serial (single-tensor) remap step:
 * detection, labeling, moments, persistence tracking, adjacency, recoloring, and
 * in-place remap of eta. Periodicity is handled through options.wrap. For
 * distributed operation use the individual building blocks (see GrainTracker).
 */
RemapResult runRemapStep(torch::Tensor & eta,
                         const GrainRemapOptions & options,
                         const std::vector<GrainMeta> & previous_grains);

} // namespace GrainRemap
