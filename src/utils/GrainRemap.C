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
#include <set>

namespace GrainRemap
{
namespace
{

constexpr double pi = 3.14159265358979323846;

std::vector<std::array<int, 3>>
neighborOffsets(int spatial_dim, int connectivity)
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
        offsets.push_back({dy, dx, 0});
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

/// result[i] = t[i + off] along dim; cells shifted in from outside hold `fill`,
/// or wrap around periodically.
torch::Tensor
shiftTensor(const torch::Tensor & t, int dim, int off, bool wrap, int64_t fill)
{
  if (off == 0)
    return t;
  if (wrap)
    return torch::roll(t, /*shifts=*/-off, /*dims=*/dim);

  auto result = torch::full_like(t, fill);
  const int64_t size = t.size(dim);
  const int64_t n = size - std::abs(off);
  if (n <= 0)
    return result;
  if (off > 0)
    result.narrow(dim, 0, n).copy_(t.narrow(dim, off, n));
  else
    result.narrow(dim, -off, n).copy_(t.narrow(dim, 0, n));
  return result;
}

/// shifted view for a multi-dimensional offset
torch::Tensor
shiftTensorOffset(const torch::Tensor & t,
                  const std::array<int, 3> & off,
                  int spatial_dim,
                  const std::array<bool, 3> & wrap,
                  int64_t fill)
{
  auto result = t;
  for (int d = 0; d < spatial_dim; ++d)
    if (off[d] != 0)
      result = shiftTensor(result, d, off[d], wrap[d], fill);
  return result;
}

/// unique sorted values of a 1D int64 tensor (handles the MPS missing-op fallback)
torch::Tensor
uniqueInt64(const torch::Tensor & values)
{
  auto v = values;
  if (v.is_mps())
    v = v.to(torch::kCPU);
  return std::get<0>(torch::_unique(v, /*sorted=*/true, /*return_inverse=*/false));
}

void
collectUniquePairs(const torch::Tensor & a_in,
                   const torch::Tensor & b_in,
                   std::vector<std::pair<int64_t, int64_t>> & pairs)
{
  const auto a = a_in.to(torch::kCPU).contiguous();
  const auto b = b_in.to(torch::kCPU).contiguous();
  const auto * pa = a.data_ptr<int64_t>();
  const auto * pb = b.data_ptr<int64_t>();
  for (int64_t i = 0; i < a.numel(); ++i)
  {
    const auto lo = std::min(pa[i], pb[i]);
    const auto hi = std::max(pa[i], pb[i]);
    if (lo != hi)
      pairs.emplace_back(lo, hi);
  }
}

void
sortUniquePairs(std::vector<std::pair<int64_t, int64_t>> & pairs)
{
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
}

} // namespace

torch::Tensor
applyLabelMap(const torch::Tensor & labels, const std::vector<int64_t> & map)
{
  auto result = torch::full_like(labels, -1);
  if (map.empty())
    return result;

  auto map_cpu = torch::from_blob(const_cast<int64_t *>(map.data()),
                                  {static_cast<int64_t>(map.size())},
                                  torch::TensorOptions().dtype(torch::kInt64))
                     .clone();
  auto map_dev = map_cpu.to(labels.device());
  auto clamped = labels.clamp_min(0);
  auto mapped = map_dev.index({clamped});
  return torch::where(labels >= 0, mapped, result);
}

Geometry
Geometry::serial(const torch::Tensor & spatial_field, const std::array<bool, 3> & periodic_in)
{
  Geometry geom;
  geom.spatial_dim = static_cast<int>(spatial_field.dim());
  for (int d = 0; d < geom.spatial_dim; ++d)
  {
    geom.owned[d] = spatial_field.size(d);
    geom.global[d] = spatial_field.size(d);
  }
  geom.periodic = periodic_in;
  return geom;
}

void
ComponentMoments::pack(double * out) const
{
  out[0] = volume;
  for (int d = 0; d < 3; ++d)
  {
    out[1 + d] = sum[d];
    out[4 + d] = sum_sin[d];
    out[7 + d] = sum_cos[d];
  }
}

void
ComponentMoments::unpack(const double * in)
{
  volume = in[0];
  for (int d = 0; d < 3; ++d)
  {
    sum[d] = in[1 + d];
    sum_sin[d] = in[4 + d];
    sum_cos[d] = in[7 + d];
  }
}

ComponentMoments &
ComponentMoments::operator+=(const ComponentMoments & other)
{
  volume += other.volume;
  for (int d = 0; d < 3; ++d)
  {
    sum[d] += other.sum[d];
    sum_sin[d] += other.sum_sin[d];
    sum_cos[d] += other.sum_cos[d];
  }
  return *this;
}

UnionFind::UnionFind(std::size_t n) : _parent(n), _rank(n, 0)
{
  std::iota(_parent.begin(), _parent.end(), 0);
}

std::size_t
UnionFind::find(std::size_t i)
{
  while (_parent[i] != i)
  {
    _parent[i] = _parent[_parent[i]];
    i = _parent[i];
  }
  return i;
}

void
UnionFind::unite(std::size_t a, std::size_t b)
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
  if (eta.dim() < 2)
    mooseError("eta must have at least 2 dimensions (spatial + order parameter).");

  const int64_t n_colors = eta.size(-1);
  const auto max_result = eta.max(-1);
  const torch::Tensor max_vals = std::get<0>(max_result);
  const torch::Tensor argmax = std::get<1>(max_result);
  const auto above = max_vals > threshold;

  std::vector<torch::Tensor> masks;
  masks.reserve(n_colors);
  for (int64_t c = 0; c < n_colors; ++c)
    masks.push_back((argmax == c) & above);
  return masks;
}

torch::Tensor
labelConnectedComponents(const torch::Tensor & mask,
                         int connectivity,
                         const std::array<bool, 3> & wrap)
{
  if (!mask.defined())
    mooseError("Mask tensor is undefined.");
  const int spatial_dim = static_cast<int>(mask.dim());
  const auto offsets = neighborOffsets(spatial_dim, connectivity);

  // initialize foreground with unique ids (flat index + 1), background zero
  auto labels = torch::zeros(mask.sizes(), mask.options().dtype(torch::kInt64));
  auto flat = torch::arange(mask.numel(), labels.options());
  labels = torch::where(mask, flat.view(mask.sizes()) + 1, labels);

  bool changed = true;
  while (changed)
  {
    // pull the minimal positive neighbor id (background stays zero)
    auto updated = labels.clone();
    for (const auto & off : offsets)
    {
      const auto neighbor = shiftTensorOffset(labels, off, spatial_dim, wrap, /*fill=*/0);
      updated = torch::where((neighbor > 0) & (neighbor < updated), neighbor, updated);
    }
    changed = torch::any(updated != labels).item<bool>();
    labels = std::move(updated);
  }

  if (labels.numel() == 0 || labels.max().item<int64_t>() == 0)
    return torch::full_like(labels, -1);

  // compress to contiguous ids 0..N-1, background -1
  const auto max_label = labels.max().item<int64_t>();
  auto unique = uniqueInt64(labels.view({-1}));
  unique = unique.masked_select(unique > 0);

  auto map_cpu = torch::full({max_label + 1}, -1, torch::TensorOptions().dtype(torch::kInt64));
  auto map_acc = map_cpu.accessor<int64_t, 1>();
  const auto unique_cpu = unique.to(torch::kCPU);
  const auto * unique_ptr = unique_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < unique_cpu.numel(); ++i)
    map_acc[unique_ptr[i]] = i;
  auto map_dev = map_cpu.to(labels.device());

  return torch::where(labels > 0, map_dev.index({labels}), torch::full_like(labels, -1));
}

torch::Tensor
buildGlobalContiguousLabels(const std::vector<torch::Tensor> & per_color_labels,
                            std::vector<int64_t> & offsets,
                            std::vector<int64_t> & counts)
{
  offsets.assign(per_color_labels.size(), 0);
  counts.assign(per_color_labels.size(), 0);
  if (per_color_labels.empty())
    return torch::Tensor();

  const auto & ref = per_color_labels.front();
  if (!ref.defined())
    mooseError("First per-color label tensor is undefined.");
  const auto shape = ref.sizes();

  torch::Tensor combined = torch::full(shape, -1, ref.options().dtype(torch::kInt64));
  int64_t running_offset = 0;
  for (std::size_t c = 0; c < per_color_labels.size(); ++c)
  {
    const auto & lbl = per_color_labels[c];
    offsets[c] = running_offset;
    if (!lbl.defined())
      continue;
    if (!lbl.sizes().equals(shape))
      mooseError("All per-color label tensors must have identical shapes.");

    auto lbl_i64 = lbl.to(combined.options());
    const int64_t max_lbl = lbl_i64.numel() ? lbl_i64.max().item<int64_t>() : -1;
    if (max_lbl >= 0)
    {
      combined = torch::where(lbl_i64 >= 0, lbl_i64 + running_offset, combined);
      counts[c] = max_lbl + 1;
      running_offset += counts[c];
    }
  }
  return combined;
}

torch::Tensor
expandLabels(const torch::Tensor & labels, int steps, const std::array<bool, 3> & wrap)
{
  if (!labels.defined())
    mooseError("Labels tensor is undefined.");
  if (steps <= 0)
    return labels;

  const int spatial_dim = static_cast<int>(labels.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Labels tensor must be 2D or 3D.");

  // breadth-first expansion: cells are claimed by the first label front that
  // reaches them (ties resolved towards the larger id) and never overwritten,
  // so the expanded regions form a (Chebyshev) nearest-grain partition
  auto current = labels.to(torch::kInt64);
  for (int step = 0; step < steps; ++step)
  {
    // separable per-axis max gives the full 3^d neighborhood of the step start
    auto front = current;
    for (int d = 0; d < spatial_dim; ++d)
    {
      const auto up = shiftTensor(front, d, 1, wrap[d], -1);
      const auto down = shiftTensor(front, d, -1, wrap[d], -1);
      front = torch::maximum(torch::maximum(front, up), down);
    }
    current = torch::where(current >= 0, current, front);
  }
  return current;
}

std::vector<ComponentMoments>
computeComponentMoments(const torch::Tensor & labels, int64_t n_labels, const Geometry & geom)
{
  if (!labels.defined())
    mooseError("Labels tensor is undefined.");
  const int spatial_dim = static_cast<int>(labels.dim());
  if (spatial_dim != geom.spatial_dim)
    mooseError("Labels dimension does not match geometry.");

  std::vector<ComponentMoments> moments(static_cast<std::size_t>(n_labels));
  if (n_labels == 0)
    return moments;

  // crop to the owned region
  auto owned = labels.to(torch::kInt64);
  for (int d = 0; d < spatial_dim; ++d)
    owned = owned.narrow(d, geom.pad, geom.owned[d]);
  owned = owned.contiguous();

  auto valid = (owned >= 0).reshape({-1});
  if (!valid.any().item<bool>())
    return moments;

  auto idx = owned.reshape({-1}).masked_select(valid);

  // double where supported; MPS only provides single precision
  const auto compute_dtype = labels.is_mps() ? torch::kFloat : torch::kDouble;
  const auto opts = torch::TensorOptions().dtype(compute_dtype).device(labels.device());

  auto volumes = torch::bincount(idx, {}, n_labels).to(torch::kCPU).to(torch::kDouble);

  std::array<torch::Tensor, 3> lin_sums;
  std::array<torch::Tensor, 3> sin_sums;
  std::array<torch::Tensor, 3> cos_sums;

  // global coordinates of the owned cells
  std::vector<torch::Tensor> axes;
  for (int d = 0; d < spatial_dim; ++d)
    axes.push_back(torch::arange(geom.owned[d], opts) + static_cast<double>(geom.global_begin[d]));
  auto grids = torch::meshgrid(axes, "ij");

  for (int d = 0; d < spatial_dim; ++d)
  {
    auto coords = grids[d].reshape({-1}).masked_select(valid);
    auto lin = torch::zeros({n_labels}, opts);
    lin.scatter_add_(0, idx, coords);
    lin_sums[d] = lin.to(torch::kCPU).to(torch::kDouble);

    if (geom.periodic[d] && geom.global[d] > 0)
    {
      auto theta = coords * (2.0 * pi / static_cast<double>(geom.global[d]));
      auto ss = torch::zeros({n_labels}, opts);
      auto sc = torch::zeros({n_labels}, opts);
      ss.scatter_add_(0, idx, torch::sin(theta));
      sc.scatter_add_(0, idx, torch::cos(theta));
      sin_sums[d] = ss.to(torch::kCPU).to(torch::kDouble);
      cos_sums[d] = sc.to(torch::kCPU).to(torch::kDouble);
    }
  }

  const auto * vol_ptr = volumes.data_ptr<double>();
  for (int64_t i = 0; i < n_labels; ++i)
  {
    auto & m = moments[static_cast<std::size_t>(i)];
    m.volume = vol_ptr[i];
    for (int d = 0; d < spatial_dim; ++d)
    {
      m.sum[d] = lin_sums[d].data_ptr<double>()[i];
      if (geom.periodic[d] && geom.global[d] > 0)
      {
        m.sum_sin[d] = sin_sums[d].data_ptr<double>()[i];
        m.sum_cos[d] = cos_sums[d].data_ptr<double>()[i];
      }
    }
  }
  return moments;
}

std::vector<std::pair<int64_t, int64_t>>
detectSeamEquivalences(const torch::Tensor & pre, const torch::Tensor & post, const Geometry & geom)
{
  std::vector<std::pair<int64_t, int64_t>> pairs;
  if (geom.pad <= 0)
    return pairs;
  if (!pre.sizes().equals(post.sizes()))
    mooseError("Pre- and post-exchange label grids must have identical shapes.");

  // mask covering only the halo region
  auto halo_mask = torch::ones(pre.sizes(), pre.options().dtype(torch::kBool));
  auto interior = halo_mask;
  for (int d = 0; d < geom.spatial_dim; ++d)
    interior = interior.narrow(d, geom.pad, geom.owned[d]);
  interior.fill_(false);

  auto m = halo_mask & (pre >= 0) & (post >= 0) & (pre != post);
  if (!m.any().item<bool>())
    return pairs;

  collectUniquePairs(pre.masked_select(m), post.masked_select(m), pairs);
  sortUniquePairs(pairs);
  return pairs;
}

std::vector<int64_t>
mergeLabels(int64_t n_labels,
            const std::vector<std::pair<int64_t, int64_t>> & equivalences,
            int64_t & n_grains)
{
  std::vector<int64_t> label_to_grain(static_cast<std::size_t>(n_labels), -1);
  n_grains = 0;
  if (n_labels == 0)
    return label_to_grain;

  UnionFind uf(static_cast<std::size_t>(n_labels));
  for (const auto & [a, b] : equivalences)
  {
    if (a < 0 || b < 0 || a >= n_labels || b >= n_labels)
      mooseError("Equivalence pair (", a, ", ", b, ") out of label range [0, ", n_labels, ").");
    uf.unite(static_cast<std::size_t>(a), static_cast<std::size_t>(b));
  }

  // assign grain ids in order of the smallest label in each set (deterministic)
  std::vector<int64_t> root_to_grain(static_cast<std::size_t>(n_labels), -1);
  for (int64_t l = 0; l < n_labels; ++l)
  {
    const auto root = uf.find(static_cast<std::size_t>(l));
    if (root_to_grain[root] < 0)
      root_to_grain[root] = n_grains++;
    label_to_grain[l] = root_to_grain[root];
  }
  return label_to_grain;
}

std::vector<GrainMeta>
finalizeGrains(const std::vector<ComponentMoments> & label_moments,
               const std::vector<int64_t> & label_to_grain,
               const std::vector<int> & label_color,
               int64_t n_grains,
               const Geometry & geom)
{
  if (label_moments.size() != label_to_grain.size() || label_moments.size() != label_color.size())
    mooseError("Label moments, grain map, and color map must have the same length.");

  std::vector<ComponentMoments> grain_moments(static_cast<std::size_t>(n_grains));
  std::vector<GrainMeta> grains(static_cast<std::size_t>(n_grains));
  for (auto & g : grains)
    g.color = -1;

  for (std::size_t l = 0; l < label_moments.size(); ++l)
  {
    const auto gid = label_to_grain[l];
    if (gid < 0 || gid >= n_grains)
      mooseError("Invalid grain id ", gid, " for label ", l, ".");
    grain_moments[gid] += label_moments[l];

    auto & g = grains[gid];
    if (g.color < 0)
      g.color = label_color[l];
    else if (g.color != label_color[l])
      mooseError("Merged labels disagree on the detected order parameter (grain ",
                 gid,
                 ": ",
                 g.color,
                 " vs ",
                 label_color[l],
                 ").");
  }

  for (int64_t gid = 0; gid < n_grains; ++gid)
  {
    auto & g = grains[gid];
    const auto & m = grain_moments[gid];
    g.grain_id = gid;
    g.new_color = g.color;
    g.volume = static_cast<int64_t>(std::llround(m.volume));
    if (m.volume <= 0.0)
      continue;

    for (int d = 0; d < geom.spatial_dim; ++d)
    {
      const double L = static_cast<double>(geom.global[d]);
      if (geom.periodic[d] && L > 0)
      {
        // volume-weighted circular mean; degenerates only for grains that span
        // the dimension nearly uniformly, where we fall back to the linear mean
        const double r = std::hypot(m.sum_sin[d], m.sum_cos[d]) / m.volume;
        if (r > 1e-9)
        {
          double angle = std::atan2(m.sum_sin[d], m.sum_cos[d]);
          if (angle < 0)
            angle += 2.0 * pi;
          g.centroid[d] = angle / (2.0 * pi) * L;
        }
        else
          g.centroid[d] = std::fmod(m.sum[d] / m.volume + L, L);
      }
      else
        g.centroid[d] = m.sum[d] / m.volume;
    }
  }
  return grains;
}

std::vector<int64_t>
matchPersistentGrains(const std::vector<GrainMeta> & previous,
                      std::vector<GrainMeta> & current,
                      const GrainRemapOptions & options,
                      const Geometry & geom,
                      int64_t first_new_id)
{
  std::vector<int64_t> persistent(current.size(), -1);
  if (current.empty())
    return persistent;

  const auto distance = [&](const GrainMeta & a, const GrainMeta & b)
  {
    double dist2 = 0.0;
    for (int d = 0; d < geom.spatial_dim; ++d)
    {
      double dd = std::abs(a.centroid[d] - b.centroid[d]);
      const double L = static_cast<double>(geom.global[d]);
      if (geom.periodic[d] && L > 0)
        dd = std::min(dd, L - dd);
      dist2 += dd * dd;
    }
    return std::sqrt(dist2);
  };

  const auto volume_compatible = [&](const GrainMeta & a, const GrainMeta & b)
  {
    if (options.tracking_volume_ratio <= 0)
      return true;
    const double lo = static_cast<double>(std::min(a.volume, b.volume));
    const double hi = static_cast<double>(std::max(a.volume, b.volume));
    return lo > 0 && hi / lo <= options.tracking_volume_ratio;
  };

  std::vector<int64_t> best_prev(current.size(), -1);
  std::vector<double> best_prev_dist(current.size(), std::numeric_limits<double>::max());
  std::vector<int64_t> best_curr(previous.size(), -1);
  std::vector<double> best_curr_dist(previous.size(), std::numeric_limits<double>::max());

  for (std::size_t i = 0; i < current.size(); ++i)
    for (std::size_t j = 0; j < previous.size(); ++j)
    {
      const double dist = distance(current[i], previous[j]);
      if (dist < best_prev_dist[i])
      {
        best_prev_dist[i] = dist;
        best_prev[i] = static_cast<int64_t>(j);
      }
      if (dist < best_curr_dist[j])
      {
        best_curr_dist[j] = dist;
        best_curr[j] = static_cast<int64_t>(i);
      }
    }

  int64_t next_persistent = first_new_id;
  for (const auto & g : previous)
    next_persistent = std::max(next_persistent, g.persistent_id + 1);

  for (std::size_t i = 0; i < current.size(); ++i)
  {
    const auto p = best_prev[i];
    const bool mutual = p >= 0 && best_curr[p] == static_cast<int64_t>(i);
    if (mutual && best_prev_dist[i] <= options.tracking_tolerance &&
        volume_compatible(current[i], previous[p]))
      persistent[i] =
          previous[p].persistent_id >= 0 ? previous[p].persistent_id : static_cast<int64_t>(p);
    else
      persistent[i] = next_persistent++;
    current[i].persistent_id = persistent[i];
  }

  return persistent;
}

std::vector<std::pair<int64_t, int64_t>>
extractAdjacencyPairs(const torch::Tensor & grain_ids,
                      int connectivity,
                      const std::array<bool, 3> & wrap)
{
  std::vector<std::pair<int64_t, int64_t>> pairs;
  if (!grain_ids.defined())
    mooseError("Grain id tensor is undefined.");

  const int spatial_dim = static_cast<int>(grain_ids.dim());
  const auto offsets = neighborOffsets(spatial_dim, connectivity);
  auto ids = grain_ids.to(torch::kInt64);

  for (const auto & off : offsets)
  {
    const auto neighbor = shiftTensorOffset(ids, off, spatial_dim, wrap, /*fill=*/-1);
    const auto m = (ids >= 0) & (neighbor >= 0) & (ids != neighbor);
    if (!m.any().item<bool>())
      continue;
    collectUniquePairs(ids.masked_select(m), neighbor.masked_select(m), pairs);
  }

  sortUniquePairs(pairs);
  return pairs;
}

std::vector<std::vector<int64_t>>
buildAdjacencyLists(int64_t n_grains, const std::vector<std::pair<int64_t, int64_t>> & pairs)
{
  std::vector<std::vector<int64_t>> adjacency(static_cast<std::size_t>(n_grains));
  for (const auto & [a, b] : pairs)
  {
    if (a < 0 || b < 0 || a >= n_grains || b >= n_grains)
      mooseError("Adjacency pair (", a, ", ", b, ") out of grain range [0, ", n_grains, ").");
    adjacency[a].push_back(b);
    adjacency[b].push_back(a);
  }
  return adjacency;
}

std::vector<int>
greedyRecolor(const std::vector<std::vector<int64_t>> & adjacency,
              const std::vector<int> & initial_colors,
              int n_colors,
              const std::vector<int64_t> & order,
              std::vector<int64_t> & conflicts,
              unsigned int max_passes)
{
  if (adjacency.size() != initial_colors.size())
    mooseError("Adjacency size and color vector size must match.");
  if (order.size() != adjacency.size())
    mooseError("Order vector size must match the number of grains.");

  std::vector<int> colors = initial_colors;
  conflicts.clear();

  for (unsigned int pass = 0; pass < max_passes; ++pass)
  {
    bool changed = false;
    for (const auto idx : order)
    {
      std::set<int> neighbor_colors;
      for (const auto n : adjacency[idx])
        neighbor_colors.insert(colors[n]);

      if (neighbor_colors.count(colors[idx]) == 0)
        continue;

      for (int c = 0; c < n_colors; ++c)
        if (neighbor_colors.count(c) == 0)
        {
          colors[idx] = c;
          changed = true;
          break;
        }
    }
    if (!changed)
      break;
  }

  // report remaining conflicts
  for (std::size_t idx = 0; idx < adjacency.size(); ++idx)
    for (const auto n : adjacency[idx])
      if (colors[n] == colors[idx])
      {
        conflicts.push_back(static_cast<int64_t>(idx));
        break;
      }

  return colors;
}

void
remapOrderParameters(torch::Tensor & eta,
                     const torch::Tensor & changed_grain_ids,
                     const std::vector<int> & old_colors,
                     const std::vector<int> & new_colors)
{
  if (old_colors.size() != new_colors.size())
    mooseError("Old and new color vectors must have the same length.");
  if (old_colors.empty())
    return;

  const auto n_colors = eta.size(-1);
  auto work = eta.contiguous();
  auto view = work.view({-1, n_colors});

  auto gid = changed_grain_ids.reshape({-1}).to(torch::kInt64);
  auto rows = torch::nonzero(gid >= 0).view(-1);
  if (rows.numel() == 0)
    return;

  std::vector<int64_t> co_vec(old_colors.begin(), old_colors.end());
  std::vector<int64_t> cn_vec(new_colors.begin(), new_colors.end());
  const auto cpu_i64 = torch::TensorOptions().dtype(torch::kInt64);
  auto co_map =
      torch::from_blob(co_vec.data(), {(int64_t)co_vec.size()}, cpu_i64).clone().to(eta.device());
  auto cn_map =
      torch::from_blob(cn_vec.data(), {(int64_t)cn_vec.size()}, cpu_i64).clone().to(eta.device());

  auto g = gid.index({rows});
  auto co = co_map.index({g});
  auto cn = cn_map.index({g});
  auto valid = (co >= 0) & (cn >= 0) & (co < n_colors) & (cn < n_colors) & (co != cn);
  rows = rows.index({valid});
  if (rows.numel() == 0)
    return;
  co = co.index({valid});
  cn = cn.index({valid});

  // move the grain's value: zero the old channel, max-combine into the new channel
  auto values = view.index({rows, co});
  view.index_put_({rows, co}, torch::zeros_like(values));
  auto dest = view.index({rows, cn});
  view.index_put_({rows, cn}, torch::maximum(dest, values));

  if (!work.is_same(eta))
    eta.copy_(work);
}

RemapResult
runRemapStep(torch::Tensor & eta,
             const GrainRemapOptions & options_in,
             const std::vector<GrainMeta> & previous_grains)
{
  RemapResult result;
  if (eta.dim() < 3)
    mooseError("eta must have at least two spatial dimensions plus the order parameter "
               "dimension.");

  const int spatial_dim = static_cast<int>(eta.dim()) - 1;
  const int n_colors = static_cast<int>(eta.size(-1));

  auto options = options_in;
  if (options.connectivity == 0)
    options.connectivity = GrainRemapOptions::defaultConnectivity(spatial_dim);

  Geometry geom;
  geom.spatial_dim = spatial_dim;
  for (int d = 0; d < spatial_dim; ++d)
  {
    geom.owned[d] = eta.size(d);
    geom.global[d] = eta.size(d);
  }
  geom.periodic = options.wrap;

  // detection and per-color labeling
  const auto masks = computeColorMasks(eta, options.threshold);
  std::vector<torch::Tensor> per_color_labels;
  per_color_labels.reserve(masks.size());
  for (const auto & mask : masks)
    per_color_labels.push_back(labelConnectedComponents(mask, options.connectivity, options.wrap));

  std::vector<int64_t> offsets, counts;
  auto combined = buildGlobalContiguousLabels(per_color_labels, offsets, counts);
  const int64_t n_labels = std::accumulate(counts.begin(), counts.end(), int64_t(0));

  result.grain_ids =
      combined.defined()
          ? combined
          : torch::full(eta.sizes().slice(0, spatial_dim), -1, eta.options().dtype(torch::kInt64));
  if (n_labels == 0)
    return result;

  std::vector<int> label_color(static_cast<std::size_t>(n_labels));
  for (int c = 0; c < n_colors; ++c)
    for (int64_t l = 0; l < counts[c]; ++l)
      label_color[offsets[c] + l] = c;

  // moments and grain aggregation (serial: labels are already global, wrap is
  // handled inside the labeling, so there are no seam equivalences)
  const auto moments = computeComponentMoments(combined, n_labels, geom);
  int64_t n_grains = 0;
  const auto label_to_grain = mergeLabels(n_labels, {}, n_grains);
  auto grains = finalizeGrains(moments, label_to_grain, label_color, n_grains, geom);

  // persistence tracking
  matchPersistentGrains(previous_grains, grains, options, geom);

  // grain id grid (same as labels here, but apply the map for generality)
  auto grain_ids = applyLabelMap(combined, label_to_grain);
  result.grain_ids = grain_ids;

  // adjacency on the halo-expanded grain id grid
  const auto expanded = expandLabels(grain_ids, options.halo_width, options.wrap);
  const auto pairs = extractAdjacencyPairs(expanded, options.connectivity, options.wrap);
  const auto adjacency = buildAdjacencyLists(n_grains, pairs);

  // recolor, processing large grains first
  std::vector<int> initial_colors(grains.size());
  for (std::size_t i = 0; i < grains.size(); ++i)
    initial_colors[i] = grains[i].color;
  std::vector<int64_t> order(grains.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(),
            order.end(),
            [&](int64_t a, int64_t b)
            {
              if (grains[a].volume != grains[b].volume)
                return grains[a].volume > grains[b].volume;
              return a < b;
            });
  const auto new_colors =
      greedyRecolor(adjacency, initial_colors, n_colors, order, result.conflicts);

  std::vector<int> old_color_vec(grains.size());
  std::vector<int> new_color_vec(grains.size());
  for (std::size_t i = 0; i < grains.size(); ++i)
  {
    grains[i].new_color = new_colors[i];
    old_color_vec[i] = grains[i].color;
    new_color_vec[i] = new_colors[i];
    if (new_colors[i] != grains[i].color)
      result.remapped_grains.push_back(static_cast<int64_t>(i));
  }

  // move recolored grains (only their dilated footprint is touched)
  if (!result.remapped_grains.empty())
  {
    std::vector<int64_t> changed_map(static_cast<std::size_t>(n_grains), -1);
    for (const auto gid : result.remapped_grains)
      changed_map[gid] = gid;
    const auto changed_expanded = applyLabelMap(expanded, changed_map);
    remapOrderParameters(eta, changed_expanded, old_color_vec, new_color_vec);
  }

  result.grains = std::move(grains);
  return result;
}

} // namespace GrainRemap
