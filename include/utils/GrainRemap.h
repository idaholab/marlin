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
#include <climits>
#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace GrainRemap
{

/// Options controlling remap behavior.
struct GrainRemapOptions
{
  int spatial_dim = 3;      ///< 2 or 3
  int n_colors = 0;         ///< number of order parameters
  double threshold = 0.0;   ///< phase-field threshold
  int halo_width = 1;       ///< halo (ghost) thickness in cells
  int connectivity = 6;     ///< 4/8 in 2D, 6/26 in 3D
  double tracking_tolerance =
      2.0; ///< max centroid distance for matching grains between remap steps
};

/// Host-side metadata for a connected component on a single rank.
struct ComponentMeta
{
  int rank = 0;
  int color = 0;
  int64_t local_label = -1;
  int64_t global_grain = -1;
  int64_t volume = 0;
  std::array<int64_t, 3> bbox_min{{INT64_MAX, INT64_MAX, INT64_MAX}};
  std::array<int64_t, 3> bbox_max{{INT64_MIN, INT64_MIN, INT64_MIN}};
  std::array<int64_t, 3> halo_min{{INT64_MAX, INT64_MAX, INT64_MAX}};
  std::array<int64_t, 3> halo_max{{INT64_MIN, INT64_MIN, INT64_MIN}};
  std::array<double, 3> centroid{{0.0, 0.0, 0.0}};
};

/// Host-side metadata for an aggregated grain (global feature).
struct GrainMeta
{
  int64_t grain_id = -1;     ///< grain id for this remap step
  int64_t persistent_id = -1;///< stable id across remap steps
  int old_color = -1;        ///< color before recoloring
  int new_color = -1;        ///< color after recoloring
  int64_t volume = 0;
  std::array<int64_t, 3> bbox_min{{INT64_MAX, INT64_MAX, INT64_MAX}};
  std::array<int64_t, 3> bbox_max{{INT64_MIN, INT64_MIN, INT64_MIN}};
  std::array<double, 3> centroid{{0.0, 0.0, 0.0}};
};

/// Handle packing/unpacking of component identity across MPI ranks.
struct ComponentRef
{
  int rank = 0;
  int color = 0;
  int64_t local_label = -1;

  uint64_t packed() const;
  bool operator==(const ComponentRef & other) const
  {
    return rank == other.rank && color == other.color && local_label == other.local_label;
  }
};

/// Standard disjoint set for multi-rank stitching.
class UnionFind
{
public:
  explicit UnionFind(size_t n);
  size_t find(size_t i);
  void unite(size_t a, size_t b);
  size_t size() const { return _parent.size(); }
  const std::vector<size_t> & parents() const { return _parent; }

private:
  std::vector<size_t> _parent;
  std::vector<size_t> _rank;
};

/// Per-color mask: true where eta[..., c] is maximal and above the threshold.
std::vector<torch::Tensor> computeColorMasks(const torch::Tensor & eta, double threshold);

/// Label connected components on GPU using iterative neighbor minima.
torch::Tensor labelConnectedComponents(const torch::Tensor & mask,
                                       const GrainRemapOptions & options);

/// Compute per-component metadata (volume, bbox, centroid, halos) on host.
std::vector<ComponentMeta> computeComponentMetadata(const torch::Tensor & labels,
                                                    int color,
                                                    int halo_width,
                                                    int rank);

/// Merge component metadata into grains using the provided equivalence pairs.
std::vector<GrainMeta>
mergeComponents(const std::vector<ComponentMeta> & components,
                const std::vector<std::pair<ComponentRef, ComponentRef>> & equivalences,
                int n_colors,
                std::vector<int64_t> * component_to_grain = nullptr);

/// Build grain_id_local tensor (CUDA int32) from local labels and a host mapping.
torch::Tensor
labelsToGlobalIds(const torch::Tensor & labels,
                  const std::vector<int64_t> & label_to_global,
                  const torch::TensorOptions & options);

/// Track grains across remap steps based on centroid proximity.
std::vector<int64_t> matchPersistentGrains(const std::vector<GrainMeta> & previous,
                                           std::vector<GrainMeta> & current,
                                           double tolerance);

/// Build adjacency list for grains of the same color (bounding-box and halo coarse tests).
std::vector<std::vector<int64_t>>
buildAdjacency(const std::vector<GrainMeta> & grains, int halo_width);

/// Greedy graph coloring that prefers to keep existing colors.
std::vector<int> greedyRecolor(const std::vector<std::vector<int64_t>> & adjacency,
                               const std::vector<int> & initial_colors,
                               int n_colors,
                               unsigned int max_passes = 8);

/// Remap eta in-place according to per-grain old/new colors and grain ids.
void remapOrderParameters(torch::Tensor & eta,
                          const torch::Tensor & grain_ids,
                          const torch::Tensor & old_colors,
                          const torch::Tensor & new_colors);

/// Convenience wrapper that runs the whole remap: masks -> labels -> metadata -> recolor.
struct RemapResult
{
  torch::Tensor grain_ids;          ///< device tensor [spatial dims], int32, -1 background
  std::vector<GrainMeta> grains;    ///< aggregated grain metadata
  std::vector<ComponentMeta> local_components; ///< metadata before stitching
};

RemapResult runRemapStep(torch::Tensor & eta,
                         const GrainRemapOptions & options,
                         int rank,
                         const std::function<void(torch::Tensor &, unsigned int)> & ghost_exchange,
                         const std::vector<GrainMeta> & previous_grains,
                         const std::vector<std::pair<ComponentRef, ComponentRef>> & equivalences,
                         std::vector<int> * chosen_colors = nullptr);

} // namespace GrainRemap

namespace std
{
template <>
struct hash<GrainRemap::ComponentRef>
{
  size_t operator()(const GrainRemap::ComponentRef & ref) const noexcept
  {
    return std::hash<uint64_t>()(ref.packed());
  }
};
} // namespace std
