/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainRemap.h"

#include "MooseError.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace GrainRemap
{
namespace
{

using torch::indexing::Slice;

std::vector<std::array<int, 3>> neighborOffsets(int spatial_dim, int connectivity)
{
  std::vector<std::array<int, 3>> offsets;
  if (spatial_dim == 2)
  {
    if (connectivity != 4 && connectivity != 8)
      mooseError("2D connectivity must be 4 or 8.");
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
      {
        if (dx == 0 && dy == 0)
          continue;
        if (connectivity == 4 && std::abs(dx) + std::abs(dy) != 1)
          continue;
        offsets.push_back({0, dy, dx});
      }
    return offsets;
  }

  if (spatial_dim != 3)
    mooseError("Spatial dimension must be 2 or 3.");

  if (connectivity != 6 && connectivity != 26)
    mooseError("3D connectivity must be 6 or 26.");

  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
      {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        if (connectivity == 6 && (std::abs(dx) + std::abs(dy) + std::abs(dz) != 1))
          continue;
        offsets.push_back({dz, dy, dx});
      }
  return offsets;
}

void applyNeighborMin(torch::Tensor & dest,
                      const torch::Tensor & src,
                      const std::array<int, 3> & offset,
                      int spatial_dim)
{
  std::vector<Slice> dst_idx;
  std::vector<Slice> src_idx;
  dst_idx.reserve(spatial_dim);
  src_idx.reserve(spatial_dim);

  for (int d = 0; d < spatial_dim; ++d)
  {
    const int shift = offset[d];
    const int64_t size = src.size(d);
    if (shift < 0)
    {
      dst_idx.emplace_back(-shift, size);
      src_idx.emplace_back(0, size + shift);
    }
    else if (shift > 0)
    {
      dst_idx.emplace_back(0, size - shift);
      src_idx.emplace_back(shift, size);
    }
    else
    {
      dst_idx.emplace_back();
      src_idx.emplace_back();
    }
  }

  const auto center_prev = src.index(dst_idx);
  const auto neighbor_prev = src.index(src_idx);
  const auto better = torch::where((neighbor_prev > 0) & (neighbor_prev < center_prev),
                                   neighbor_prev,
                                   center_prev);
  auto dest_slice = dest.index(dst_idx);
  dest.index_put_(dst_idx, torch::min(dest_slice, better));
}

std::array<int64_t, 3> clampBBoxMin(const std::array<int64_t, 3> & bbox, int halo, int spatial_dim)
{
  std::array<int64_t, 3> out = bbox;
  for (int d = 0; d < spatial_dim; ++d)
    out[d] = std::max<int64_t>(0, bbox[d] - halo);
  for (int d = spatial_dim; d < 3; ++d)
    out[d] = 0;
  return out;
}

std::array<int64_t, 3>
clampBBoxMax(const std::array<int64_t, 3> & bbox,
             int halo,
             const std::array<int64_t, 3> & shape,
             int spatial_dim)
{
  std::array<int64_t, 3> out = bbox;
  for (int d = 0; d < spatial_dim; ++d)
    out[d] = std::min<int64_t>(shape[d] - 1, bbox[d] + halo);
  for (int d = spatial_dim; d < 3; ++d)
    out[d] = 0;
  return out;
}

int grainColor(const GrainMeta & g)
{
  return g.new_color >= 0 ? g.new_color : g.old_color;
}

} // namespace

uint64_t
ComponentRef::packed() const
{
  return (static_cast<uint64_t>(static_cast<uint32_t>(rank)) << 48) ^
         (static_cast<uint64_t>(static_cast<uint16_t>(color)) << 32) ^
         static_cast<uint64_t>(local_label & 0xffffffffULL);
}

UnionFind::UnionFind(size_t n) : _parent(n), _rank(n, 0)
{
  std::iota(_parent.begin(), _parent.end(), 0);
}

size_t
UnionFind::find(size_t i)
{
  if (_parent[i] == i)
    return i;
  _parent[i] = find(_parent[i]);
  return _parent[i];
}

void
UnionFind::unite(size_t a, size_t b)
{
  const auto ra = find(a);
  const auto rb = find(b);
  if (ra == rb)
    return;
  if (_rank[ra] < _rank[rb])
    _parent[ra] = rb;
  else if (_rank[ra] > _rank[rb])
    _parent[rb] = ra;
  else
  {
    _parent[rb] = ra;
    _rank[ra]++;
  }
}

std::vector<torch::Tensor>
computeColorMasks(const torch::Tensor & eta, double threshold)
{
  const auto ndim = eta.dim();
  if (ndim < 2)
    mooseError("eta must have at least 2 dimensions (spatial + color).");

  const int64_t n_colors = eta.size(-1);
  // trailing color dimension, spatial dimensions leading
  const auto max_result = eta.max(-1, true);
  const torch::Tensor max_vals = std::get<0>(max_result);
  const torch::Tensor argmax = std::get<1>(max_result);

  std::vector<torch::Tensor> masks;
  masks.reserve(n_colors);
  for (int64_t c = 0; c < n_colors; ++c)
  {
    auto mask = (argmax == c) & (eta.select(-1, c) > threshold) & (max_vals.squeeze(-1) > threshold);
    masks.push_back(mask);
  }
  return masks;
}

torch::Tensor
labelConnectedComponents(const torch::Tensor & mask, const GrainRemapOptions & options)
{
  if (!mask.defined())
    mooseError("Mask tensor is undefined.");
  const int spatial_dim = static_cast<int>(mask.dim());
  const auto offsets = neighborOffsets(spatial_dim, options.connectivity);

  auto labels = torch::zeros(mask.sizes(), mask.options().dtype(torch::kInt64));
  // initialize foreground with unique ids, background zero
  auto flat = torch::arange(mask.numel(), labels.options());
  labels = torch::where(mask, flat.view(mask.sizes()) + 1, labels);

  bool changed = true;
  while (changed)
  {
    // Bellman–Ford style propagation: pull minimal non-zero neighbor id
    auto updated = labels.clone();
    for (const auto & off : offsets)
      applyNeighborMin(updated, labels, off, spatial_dim);
    changed = torch::any(updated != labels).item<bool>();
    labels = std::move(updated);
  }

  if (labels.numel() == 0)
    return labels;

  const auto max_label = labels.max().item<int64_t>();
  if (max_label == 0)
    return torch::full_like(labels, -1);

  auto unique = torch::unique(labels.view({-1}));
  unique = unique.masked_select(unique > 0);
  if (unique.numel() == 0)
    return torch::full_like(labels, -1);

  auto map_cpu = torch::full({max_label + 1}, -1, labels.options().device(torch::kCPU));
  const auto unique_cpu = unique.to(torch::kCPU());
  auto map_acc = map_cpu.accessor<int64_t, 1>();
  const auto * unique_ptr = unique_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < unique_cpu.numel(); ++i)
    map_acc[unique_ptr[i]] = i;
  auto map_device = map_cpu.to(labels.device());

  auto mapped =
      torch::where(labels > 0, map_device.index({labels}), torch::full_like(labels, -1, labels.options()));
  return mapped;
}

std::vector<ComponentMeta>
computeComponentMetadata(const torch::Tensor & labels, int color, int halo_width, int rank)
{
  if (!labels.defined())
    mooseError("Labels tensor is undefined.");
  const int spatial_dim = static_cast<int>(labels.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Labels tensor must be 2D or 3D.");

  const auto labels_cpu = labels.to(torch::kCPU());
  const auto shape = labels_cpu.sizes();
  const int64_t nx = spatial_dim == 3 ? shape[2] : shape[1];
  const int64_t ny = spatial_dim == 3 ? shape[1] : shape[0];
  const int64_t nz = spatial_dim == 3 ? shape[0] : 1;

  const int64_t max_label = labels.numel() ? labels.max().item<int64_t>() : -1;
  if (max_label < 0)
    return {};
  const size_t n_comp = static_cast<size_t>(max_label + 1);

  std::vector<ComponentMeta> meta(n_comp);
  std::vector<std::array<double, 3>> coord_sums(n_comp, {0.0, 0.0, 0.0});
  for (size_t i = 0; i < n_comp; ++i)
  {
    meta[i].color = color;
    meta[i].rank = rank;
    meta[i].local_label = static_cast<int64_t>(i);
  }

  const auto * data_ptr = labels_cpu.data_ptr<int64_t>();
  const int64_t total = labels_cpu.numel();
  for (int64_t idx = 0; idx < total; ++idx)
  {
    const int64_t lbl = data_ptr[idx];
    if (lbl < 0)
      continue;

    const int64_t x = idx % nx;
    const int64_t y = spatial_dim == 3 ? (idx / nx) % ny : idx / nx;
    const int64_t z = spatial_dim == 3 ? idx / (nx * ny) : 0;

    auto & m = meta[lbl];
    m.volume++;
    m.bbox_min[0] = std::min<int64_t>(m.bbox_min[0], z);
    m.bbox_min[1] = std::min<int64_t>(m.bbox_min[1], y);
    m.bbox_min[2] = std::min<int64_t>(m.bbox_min[2], x);
    m.bbox_max[0] = std::max<int64_t>(m.bbox_max[0], z);
    m.bbox_max[1] = std::max<int64_t>(m.bbox_max[1], y);
    m.bbox_max[2] = std::max<int64_t>(m.bbox_max[2], x);

    coord_sums[lbl][0] += static_cast<double>(z);
    coord_sums[lbl][1] += static_cast<double>(y);
    coord_sums[lbl][2] += static_cast<double>(x);
  }

  const std::array<int64_t, 3> full_shape{{nz, ny, nx}};
  for (size_t i = 0; i < n_comp; ++i)
  {
    auto & m = meta[i];
    if (m.volume > 0)
    {
      m.centroid[0] = coord_sums[i][0] / static_cast<double>(m.volume);
      m.centroid[1] = coord_sums[i][1] / static_cast<double>(m.volume);
      m.centroid[2] = coord_sums[i][2] / static_cast<double>(m.volume);
      m.halo_min = clampBBoxMin(m.bbox_min, halo_width, spatial_dim);
      m.halo_max = clampBBoxMax(m.bbox_max, halo_width, full_shape, spatial_dim);
    }
  }

  return meta;
}

std::vector<GrainMeta>
mergeComponents(const std::vector<ComponentMeta> & components,
                const std::vector<std::pair<ComponentRef, ComponentRef>> & equivalences,
                int n_colors,
                std::vector<int64_t> * component_to_grain)
{
  (void)n_colors;
  if (components.empty())
  {
    if (component_to_grain)
      component_to_grain->clear();
    return {};
  }

  std::unordered_map<ComponentRef, size_t> ref_to_idx;
  ref_to_idx.reserve(components.size());
  for (size_t i = 0; i < components.size(); ++i)
  {
    const ComponentRef ref{components[i].rank, components[i].color, components[i].local_label};
    ref_to_idx[ref] = i;
  }

  UnionFind uf(components.size());
  for (const auto & eq : equivalences)
  {
    auto it_a = ref_to_idx.find(eq.first);
    auto it_b = ref_to_idx.find(eq.second);
    if (it_a != ref_to_idx.end() && it_b != ref_to_idx.end())
      uf.unite(it_a->second, it_b->second);
  }

  std::unordered_map<size_t, int64_t> root_to_gid;
  std::vector<int64_t> comp_to_gid(components.size(), -1);
  for (size_t i = 0; i < components.size(); ++i)
  {
    const auto root = uf.find(i);
    auto it = root_to_gid.find(root);
    if (it == root_to_gid.end())
    {
      const int64_t gid = static_cast<int64_t>(root_to_gid.size());
      root_to_gid[root] = gid;
      comp_to_gid[i] = gid;
    }
    else
      comp_to_gid[i] = it->second;
  }

  std::vector<GrainMeta> grains(root_to_gid.size());
  std::vector<std::array<double, 3>> weighted_centroid(grains.size(), {0.0, 0.0, 0.0});
  for (size_t i = 0; i < components.size(); ++i)
  {
    const int64_t gid = comp_to_gid[i];
    auto & g = grains[gid];
    const auto & c = components[i];

    if (g.grain_id < 0)
    {
      g.grain_id = gid;
      g.old_color = c.color;
      g.new_color = c.color;
      g.bbox_min = c.bbox_min;
      g.bbox_max = c.bbox_max;
    }
    else
    {
      g.bbox_min[0] = std::min(g.bbox_min[0], c.bbox_min[0]);
      g.bbox_min[1] = std::min(g.bbox_min[1], c.bbox_min[1]);
      g.bbox_min[2] = std::min(g.bbox_min[2], c.bbox_min[2]);
      g.bbox_max[0] = std::max(g.bbox_max[0], c.bbox_max[0]);
      g.bbox_max[1] = std::max(g.bbox_max[1], c.bbox_max[1]);
      g.bbox_max[2] = std::max(g.bbox_max[2], c.bbox_max[2]);
    }

    g.volume += c.volume;
    weighted_centroid[gid][0] += c.centroid[0] * static_cast<double>(c.volume);
    weighted_centroid[gid][1] += c.centroid[1] * static_cast<double>(c.volume);
    weighted_centroid[gid][2] += c.centroid[2] * static_cast<double>(c.volume);
  }

  for (size_t i = 0; i < grains.size(); ++i)
  {
    auto & g = grains[i];
    if (g.volume > 0)
    {
      g.centroid[0] = weighted_centroid[i][0] / static_cast<double>(g.volume);
      g.centroid[1] = weighted_centroid[i][1] / static_cast<double>(g.volume);
      g.centroid[2] = weighted_centroid[i][2] / static_cast<double>(g.volume);
    }
    g.grain_id = static_cast<int64_t>(i);
    g.old_color = std::max(0, g.old_color);
    g.new_color = std::max(0, g.new_color);
  }

  if (component_to_grain)
    *component_to_grain = std::move(comp_to_gid);

  return grains;
}

torch::Tensor
labelsToGlobalIds(const torch::Tensor & labels,
                  const std::vector<int64_t> & label_to_global,
                  const torch::TensorOptions & options)
{
  auto result = torch::full(labels.sizes(), -1, options);
  if (label_to_global.empty())
    return result;

  auto map_cpu =
      torch::from_blob(const_cast<int64_t *>(label_to_global.data()),
                       {static_cast<int64_t>(label_to_global.size())},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();
  auto map = map_cpu.to(labels.device());
  auto labels_pos = torch::where(labels >= 0, labels, torch::zeros_like(labels));
  auto gathered = map.index({labels_pos});
  result = torch::where(labels >= 0, gathered.to(options.dtype()), result);
  return result;
}

std::vector<int64_t>
matchPersistentGrains(const std::vector<GrainMeta> & previous,
                      std::vector<GrainMeta> & current,
                      double tolerance)
{
  std::vector<int64_t> persistent(current.size(), -1);
  if (current.empty())
    return persistent;

  std::vector<int64_t> best_prev_for_curr(current.size(), -1);
  std::vector<double> best_prev_dist(current.size(),
                                     std::numeric_limits<double>::max());
  for (size_t i = 0; i < current.size(); ++i)
  {
    for (size_t j = 0; j < previous.size(); ++j)
    {
      double dz = current[i].centroid[0] - previous[j].centroid[0];
      double dy = current[i].centroid[1] - previous[j].centroid[1];
      double dx = current[i].centroid[2] - previous[j].centroid[2];
      const double dist = std::sqrt(dz * dz + dy * dy + dx * dx);
      if (dist < best_prev_dist[i])
      {
        best_prev_dist[i] = dist;
        best_prev_for_curr[i] = static_cast<int64_t>(j);
      }
    }
  }

  std::vector<int64_t> best_curr_for_prev(previous.size(), -1);
  std::vector<double> best_curr_dist(previous.size(),
                                     std::numeric_limits<double>::max());
  for (size_t j = 0; j < previous.size(); ++j)
  {
    for (size_t i = 0; i < current.size(); ++i)
    {
      double dz = current[i].centroid[0] - previous[j].centroid[0];
      double dy = current[i].centroid[1] - previous[j].centroid[1];
      double dx = current[i].centroid[2] - previous[j].centroid[2];
      const double dist = std::sqrt(dz * dz + dy * dy + dx * dx);
      if (dist < best_curr_dist[j])
      {
        best_curr_dist[j] = dist;
        best_curr_for_prev[j] = static_cast<int64_t>(i);
      }
    }
  }

  int64_t next_persistent = 0;
  for (const auto & g : previous)
    next_persistent = std::max(next_persistent, g.persistent_id + 1);

  for (size_t i = 0; i < current.size(); ++i)
  {
    const int64_t p = best_prev_for_curr[i];
    const bool mutual = (p >= 0 && best_curr_for_prev[p] == static_cast<int64_t>(i));
    if (mutual && best_prev_dist[i] <= tolerance)
    {
      persistent[i] = previous[p].persistent_id >= 0 ? previous[p].persistent_id : p;
      current[i].persistent_id = persistent[i];
      current[i].old_color = previous[p].new_color;
      current[i].new_color = previous[p].new_color;
    }
    else
    {
      persistent[i] = next_persistent++;
      current[i].persistent_id = persistent[i];
      if (current[i].old_color < 0)
        current[i].old_color = 0;
      if (current[i].new_color < 0)
        current[i].new_color = current[i].old_color;
    }
  }

  return persistent;
}

std::vector<std::vector<int64_t>>
buildAdjacency(const std::vector<GrainMeta> & grains, int halo_width)
{
  std::set<std::pair<int64_t, int64_t>> edges;
  const size_t n = grains.size();
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
    {
      const int color_i = grainColor(grains[i]);
      const int color_j = grainColor(grains[j]);
      if (color_i < 0 || color_j < 0 || color_i != color_j)
        continue;

      bool overlap = true;
      for (int d = 0; d < 3; ++d)
      {
        const int64_t a_min = grains[i].bbox_min[d] - halo_width;
        const int64_t a_max = grains[i].bbox_max[d] + halo_width;
        const int64_t b_min = grains[j].bbox_min[d] - halo_width;
        const int64_t b_max = grains[j].bbox_max[d] + halo_width;
        if (a_max < b_min || b_max < a_min)
        {
          overlap = false;
          break;
        }
      }
      if (overlap)
        edges.emplace(static_cast<int64_t>(i), static_cast<int64_t>(j));
    }

  std::vector<std::vector<int64_t>> adj(n);
  for (const auto & e : edges)
  {
    adj[e.first].push_back(e.second);
    adj[e.second].push_back(e.first);
  }
  return adj;
}

std::vector<int>
greedyRecolor(const std::vector<std::vector<int64_t>> & adjacency,
              const std::vector<int> & initial_colors,
              int n_colors,
              unsigned int max_passes)
{
  if (adjacency.size() != initial_colors.size())
    mooseError("Adjacency size and color vector size must match.");

  std::vector<int> colors = initial_colors;
  std::vector<size_t> order(colors.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  for (unsigned int pass = 0; pass < max_passes; ++pass)
  {
    std::shuffle(order.begin(), order.end(), rng);
    bool changed = false;
    for (const auto idx : order)
    {
      std::unordered_set<int> neighbor_colors;
      for (const auto n : adjacency[idx])
        if (n >= 0 && static_cast<size_t>(n) < colors.size())
          neighbor_colors.insert(colors[n]);

      const int current = colors[idx];
      if (neighbor_colors.find(current) == neighbor_colors.end())
        continue;

      int replacement = current;
      for (int c = 0; c < n_colors; ++c)
        if (neighbor_colors.find(c) == neighbor_colors.end())
        {
          replacement = c;
          break;
        }
      if (replacement != current)
      {
        colors[idx] = replacement;
        changed = true;
      }
    }
    if (!changed)
      break;
  }

  return colors;
}

void
remapOrderParameters(torch::Tensor & eta,
                     const torch::Tensor & grain_ids,
                     const torch::Tensor & old_colors,
                     const torch::Tensor & new_colors)
{
  const auto n_colors = eta.size(-1);
  // spatial dims are flattened; color dimension stays trailing
  auto eta_old = eta.clone();
  eta.zero_();

  auto eta_old_view = eta_old.view({-1, n_colors});
  auto eta_view = eta.view({-1, n_colors});

  auto gid_flat = grain_ids.view({-1}).to(torch::kLong);
  auto mask = gid_flat >= 0;
  auto rows = torch::nonzero(mask).view(-1);
  if (rows.numel() == 0)
    return;

  auto gid_rows = gid_flat.index({rows});
  auto co = old_colors.index({gid_rows});
  auto cn = new_colors.index({gid_rows});
  auto valid =
      (co >= 0) & (cn >= 0) & (co < n_colors) & (cn < n_colors);
  rows = rows.index({valid});
  if (rows.numel() == 0)
    return;

  co = co.index({valid});
  cn = cn.index({valid});

  auto values = eta_old_view.index({rows, co});
  eta_view.index_put_({rows, cn}, values);
}

RemapResult
runRemapStep(torch::Tensor & eta,
             const GrainRemapOptions & options,
             int rank,
             const std::function<void(torch::Tensor &, unsigned int)> & ghost_exchange,
             const std::vector<GrainMeta> & previous_grains,
             const std::vector<std::pair<ComponentRef, ComponentRef>> & equivalences,
             std::vector<int> * chosen_colors)
{
  RemapResult result;
  auto masks = computeColorMasks(eta, options.threshold);
  if (masks.empty())
    return result;

  std::vector<torch::Tensor> labels_per_color;
  labels_per_color.reserve(masks.size());

  for (size_t c = 0; c < masks.size(); ++c)
  {
    auto labels = labelConnectedComponents(masks[c], options);
    if (ghost_exchange)
      // caller provides halo exchange for arbitrary tensors
      ghost_exchange(labels, options.halo_width);
    auto meta = computeComponentMetadata(labels, static_cast<int>(c), options.halo_width, rank);
    result.local_components.insert(result.local_components.end(), meta.begin(), meta.end());
    labels_per_color.push_back(labels);
  }

  std::vector<int64_t> comp_to_grain;
  auto grains =
      mergeComponents(result.local_components, equivalences, options.n_colors, &comp_to_grain);

  // build per-color label -> grain mapping
  std::vector<std::vector<int64_t>> per_color_map(options.n_colors);
  for (size_t i = 0; i < result.local_components.size(); ++i)
  {
    const auto & comp = result.local_components[i];
    if (comp.local_label < 0)
      continue;
    auto & table = per_color_map[comp.color];
    if (static_cast<size_t>(comp.local_label) >= table.size())
      table.resize(static_cast<size_t>(comp.local_label) + 1, -1);
    table[comp.local_label] = comp_to_grain[i];
  }

  auto gid_local =
      torch::full(labels_per_color.front().sizes(),
                  -1,
                  torch::TensorOptions().device(eta.device()).dtype(torch::kInt32));
  for (size_t c = 0; c < labels_per_color.size(); ++c)
  {
    const auto mapped = labelsToGlobalIds(labels_per_color[c],
                                          per_color_map[c],
                                          gid_local.options());
    gid_local = torch::where(mapped >= 0, mapped.to(gid_local.dtype()), gid_local);
  }

  // track persistence and colors
  auto persistent_ids = matchPersistentGrains(previous_grains, grains, options.tracking_tolerance);
  std::vector<int> initial_colors;
  initial_colors.reserve(grains.size());
  for (const auto & g : grains)
    initial_colors.push_back(grainColor(g));

  const auto adjacency = buildAdjacency(grains, options.halo_width);
  auto new_colors = greedyRecolor(adjacency, initial_colors, options.n_colors);
  for (size_t i = 0; i < grains.size(); ++i)
    grains[i].new_color = new_colors[i];

  std::vector<int64_t> old_color_vec(grains.size(), -1);
  std::vector<int64_t> new_color_vec(grains.size(), -1);
  for (size_t i = 0; i < grains.size(); ++i)
  {
    old_color_vec[i] = grains[i].old_color;
    new_color_vec[i] = grains[i].new_color;
  }

  auto old_color_t = torch::from_blob(old_color_vec.data(),
                                      {static_cast<int64_t>(old_color_vec.size())},
                                      torch::TensorOptions().dtype(torch::kInt64))
                         .clone()
                         .to(eta.device());
  auto new_color_t = torch::from_blob(new_color_vec.data(),
                                      {static_cast<int64_t>(new_color_vec.size())},
                                      torch::TensorOptions().dtype(torch::kInt64))
                         .clone()
                         .to(eta.device());

  remapOrderParameters(eta, gid_local.to(torch::kLong), old_color_t, new_color_t);

  result.grain_ids = gid_local;
  result.grains = std::move(grains);
  if (chosen_colors)
    *chosen_colors = std::move(new_colors);
  return result;
}

} // namespace GrainRemap
