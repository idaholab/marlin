/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainRemap.h"

#include "MooseError.h"
#include "MarlinUtils.h"
#include "PetscSupport.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

namespace GrainRemap
{
namespace
{

using torch::indexing::Slice;
using torch::indexing::TensorIndex;

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

  std::vector<TensorIndex> dst_ti(dst_idx.begin(), dst_idx.end());
  std::vector<TensorIndex> src_ti(src_idx.begin(), src_idx.end());

  const auto center_prev = src.index(dst_ti);
  const auto neighbor_prev = src.index(src_ti);
  const auto better = torch::where((neighbor_prev > 0) & (neighbor_prev < center_prev),
                                   neighbor_prev,
                                   center_prev);
  auto dest_slice = dest.index(dst_ti);
  dest.index_put_(dst_ti, torch::min(dest_slice, better));
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
  const auto max_result = eta.max(-1);
  const torch::Tensor max_vals = std::get<0>(max_result);
  const torch::Tensor argmax = std::get<1>(max_result);

  std::vector<torch::Tensor> masks;
  masks.reserve(n_colors);
  for (int64_t c = 0; c < n_colors; ++c)
  {
    auto mask = (argmax == c) & (eta.select(-1, c) > threshold) & (max_vals > threshold);
    masks.push_back(mask);
  }
  return masks;
}

std::pair<torch::Tensor, torch::Tensor>
labelConnectedComponentsWithRaw(const torch::Tensor & mask, const GrainRemapOptions & options)
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
    return {labels, labels};

  const auto max_label = labels.max().item<int64_t>();
  if (max_label == 0)
    return {labels, torch::full_like(labels, -1)};

  const bool device_needs_cpu_unique = labels.is_mps();
  auto labels_flat = labels.view({-1});
  auto unique_tuple = torch::_unique(
      device_needs_cpu_unique ? labels_flat.to(torch::Device(torch::kCPU)) : labels_flat,
      /*sorted=*/true,
      /*return_inverse=*/false);
  auto unique = std::get<0>(unique_tuple);
  unique = unique.masked_select(unique > 0);
  if (unique.numel() == 0)
    return {labels, torch::full_like(labels, -1)};

  auto map_cpu = torch::full({max_label + 1}, -1, labels.options().device(torch::kCPU));
  const auto unique_cpu = unique.to(torch::Device(torch::kCPU));
  auto map_acc = map_cpu.accessor<int64_t, 1>();
  const auto * unique_ptr = unique_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < unique_cpu.numel(); ++i)
    map_acc[unique_ptr[i]] = i;
  auto map_device = map_cpu.to(labels.device());

  auto mapped =
      torch::where(labels > 0, map_device.index({labels}), torch::full_like(labels, -1, labels.options()));
  return {labels, mapped};
}

torch::Tensor
labelConnectedComponents(const torch::Tensor & mask, const GrainRemapOptions & options)
{
  return labelConnectedComponentsWithRaw(mask, options).second;
}

torch::Tensor
combineRawLabelsAcrossColors(const std::vector<torch::Tensor> & raw_labels)
{
  if (raw_labels.empty())
    return torch::Tensor();

  const auto & ref = raw_labels.front();
  if (!ref.defined())
    mooseError("First raw label tensor is undefined.");
  const auto shape = ref.sizes();

  torch::Tensor result = torch::zeros(shape, ref.options().dtype(torch::kInt64));
  int64_t offset = 0;
  for (const auto & lbl : raw_labels)
  {
    if (!lbl.defined())
      continue;
    if (!lbl.sizes().equals(shape))
      mooseError("All raw label tensors must have identical shapes.");

    auto lbl_i64 = lbl.to(result.options());
    const int64_t max_lbl = lbl_i64.numel() ? lbl_i64.max().item<int64_t>() : 0;
    if (max_lbl <= 0)
      continue;
    result = torch::where(lbl_i64 > 0, lbl_i64 + offset, result);
    offset += max_lbl;
  }
  return result;
}

torch::Tensor
buildGlobalContiguousLabels(const std::vector<torch::Tensor> & per_color_labels,
                            std::vector<int64_t> & offsets)
{
  if (per_color_labels.empty())
    return torch::Tensor();

  const auto & ref = per_color_labels.front();
  if (!ref.defined())
    mooseError("First per-color label tensor is undefined.");
  const auto shape = ref.sizes();
  offsets.assign(per_color_labels.size(), 0);

  torch::Tensor combined = torch::full(shape, -1, ref.options().dtype(torch::kInt64));
  int64_t running_offset = 0;
  for (size_t i = 0; i < per_color_labels.size(); ++i)
  {
    const auto & lbl = per_color_labels[i];
    if (!lbl.defined())
      continue;
    if (!lbl.sizes().equals(shape))
      mooseError("All per-color label tensors must have identical shapes.");

    // Expect background -1 and foreground 0..Nc-1
    auto lbl_i64 = lbl.to(combined.options());
    const int64_t max_lbl = lbl_i64.numel() ? lbl_i64.max().item<int64_t>() : -1;
    if (max_lbl >= 0)
    {
      combined = torch::where(lbl_i64 >= 0, lbl_i64 + running_offset, combined);
      offsets[i] = running_offset;
      running_offset += max_lbl + 1;
    }
    else
      offsets[i] = running_offset;
  }
  return combined;
}

torch::Tensor
dilateMask(const torch::Tensor & mask, int halo_width)
{
  if (!mask.defined())
    mooseError("Mask tensor is undefined.");
  if (halo_width <= 0)
    return mask.to(torch::kBool);

  const int spatial_dim = static_cast<int>(mask.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Mask must be 2D or 3D for dilation.");

  auto base = mask.to(torch::kBool);
  auto result = base.clone();

  if (spatial_dim == 2)
  {
    for (int dy = -halo_width; dy <= halo_width; ++dy)
      for (int dx = -halo_width; dx <= halo_width; ++dx)
      {
        if (dy == 0 && dx == 0)
          continue;

        std::vector<Slice> dst_idx;
        std::vector<Slice> src_idx;
        dst_idx.reserve(2);
        src_idx.reserve(2);

        const int shifts[2] = {dy, dx};
        for (int d = 0; d < 2; ++d)
        {
          const int shift = shifts[d];
          const int64_t size = base.size(d);
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

        std::vector<TensorIndex> dst_ti(dst_idx.begin(), dst_idx.end());
        std::vector<TensorIndex> src_ti(src_idx.begin(), src_idx.end());
        auto dst_view = result.index(dst_ti);
        auto src_view = base.index(src_ti);
        result.index_put_(dst_ti, dst_view | src_view);
      }
  }
  else
  {
    for (int dz = -halo_width; dz <= halo_width; ++dz)
      for (int dy = -halo_width; dy <= halo_width; ++dy)
        for (int dx = -halo_width; dx <= halo_width; ++dx)
        {
          if (dz == 0 && dy == 0 && dx == 0)
            continue;

          std::vector<Slice> dst_idx;
          std::vector<Slice> src_idx;
          dst_idx.reserve(3);
          src_idx.reserve(3);

          const int shifts[3] = {dz, dy, dx};
          for (int d = 0; d < 3; ++d)
          {
            const int shift = shifts[d];
            const int64_t size = base.size(d);
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

          std::vector<TensorIndex> dst_ti(dst_idx.begin(), dst_idx.end());
          std::vector<TensorIndex> src_ti(src_idx.begin(), src_idx.end());
          auto dst_view = result.index(dst_ti);
          auto src_view = base.index(src_ti);
          result.index_put_(dst_ti, dst_view | src_view);
        }
  }

  return result;
}

torch::Tensor
expandLabelsWithHalo(const torch::Tensor & labels, int halo_width)
{
  if (!labels.defined())
    mooseError("Labels tensor is undefined.");
  if (halo_width <= 0)
    return labels;

  const int spatial_dim = static_cast<int>(labels.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Labels tensor must be 2D or 3D.");

  // Shift valid labels by +1 so label 0 is preserved through pooling; background stays 0.
  auto shifted =
      torch::where(labels >= 0, labels.to(torch::kInt64) + 1, torch::zeros_like(labels, torch::kInt64));
  auto current = shifted;
  for (int step = 0; step < halo_width; ++step)
  {
    if (spatial_dim == 2)
    {
      auto input = current.unsqueeze(0).unsqueeze(0); // NCHW
      auto pooled = torch::nn::functional::max_pool2d(
          input, torch::nn::functional::MaxPool2dFuncOptions(3).stride(1).padding(1));
      current = pooled.squeeze();
    }
    else
    {
      auto input = current.unsqueeze(0).unsqueeze(0); // NCDHW
      auto pooled = torch::nn::functional::max_pool3d(
          input, torch::nn::functional::MaxPool3dFuncOptions(3).stride(1).padding(1));
      current = pooled.squeeze();
    }
  }
  // Undo the shift: background -> -1, labels -> label-1.
  auto out = torch::where(current > 0, current - 1, torch::full_like(current, -1));
  return out;
}

AdjacencyBuildResult
buildHaloAdjacency(const torch::Tensor & halo_labels, int connectivity)
{
  if (!halo_labels.defined())
    mooseError("Halo labels tensor is undefined.");

  const int spatial_dim = static_cast<int>(halo_labels.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Halo labels tensor must be 2D or 3D.");

  if ((spatial_dim == 2 && connectivity != 4 && connectivity != 8) ||
      (spatial_dim == 3 && connectivity != 6 && connectivity != 26))
    mooseError("Unsupported connectivity for halo adjacency.");

  AdjacencyBuildResult result;
  const bool on_device = !halo_labels.device().is_cpu();

  if (!on_device)
  {
    auto labels_cpu = halo_labels.to(torch::kCPU).to(torch::kInt64);
    auto valid = labels_cpu >= 0;
    auto uniq = torch::_unique(labels_cpu.masked_select(valid), /*sorted=*/true, /*return_inverse=*/false);
    result.unique_labels = std::get<0>(uniq);
    const int64_t n_lbl = result.unique_labels.numel();
    if (n_lbl == 0)
    {
      result.adjacency =
          torch::zeros({0, 0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
      return result;
    }

    const int64_t max_lbl = labels_cpu.max().item<int64_t>();
    std::vector<int64_t> map(static_cast<size_t>(max_lbl + 1), -1);
    const auto * uniq_ptr = result.unique_labels.data_ptr<int64_t>();
    for (int64_t i = 0; i < n_lbl; ++i)
      map[uniq_ptr[i]] = i;

    result.adjacency =
        torch::zeros({n_lbl, n_lbl}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const auto adj_stride = static_cast<int64_t>(result.adjacency.stride(0));
    auto * adj_ptr = result.adjacency.data_ptr<int64_t>();

    const auto sizes = labels_cpu.sizes();
    const int64_t D0 = sizes[0];
    const int64_t D1 = sizes[1];
    const int64_t D2 = (spatial_dim == 3) ? sizes[2] : 1;
    const auto * data = labels_cpu.data_ptr<int64_t>();

    auto offsets_raw = neighborOffsets(spatial_dim, connectivity);
    std::vector<std::array<int, 3>> offsets;
    offsets.reserve(offsets_raw.size());
    if (spatial_dim == 2)
      for (const auto & o : offsets_raw)
        offsets.push_back({0, o[0], o[1]}); // map (dy,dx) -> (z=0,y,x)
    else
      offsets = std::move(offsets_raw);
    for (int64_t z = 0; z < D2; ++z)
      for (int64_t y = 0; y < D0; ++y)
        for (int64_t x = 0; x < D1; ++x)
        {
          const int64_t lbl = (spatial_dim == 3) ? data[(z * D0 + y) * D1 + x] : data[y * D1 + x];
          if (lbl < 0)
            continue;
          const int64_t mi = map[lbl];
          for (const auto & off : offsets)
          {
            const int64_t zz = z + off[0];
            const int64_t yy = y + off[1];
            const int64_t xx = x + off[2];
            if (zz < 0 || zz >= D2 || yy < 0 || yy >= D0 || xx < 0 || xx >= D1)
              continue;
            const int64_t lbl2 =
                (spatial_dim == 3) ? data[(zz * D0 + yy) * D1 + xx] : data[yy * D1 + xx];
            if (lbl2 >= 0 && lbl2 != lbl)
            {
              const int64_t mj = map[lbl2];
              adj_ptr[mi * adj_stride + mj] = 1;
              adj_ptr[mj * adj_stride + mi] = 1;
            }
          }
        }
    return result;
  }

  // Device path: generate neighbor pairs on device, then build adjacency on CPU from pairs.
  auto labels = halo_labels.to(torch::kInt64);
  const auto offsets = neighborOffsets(spatial_dim, connectivity);
  std::vector<torch::Tensor> edge_pairs;
  edge_pairs.reserve(offsets.size());

  for (const auto & off : offsets)
  {
    std::vector<TensorIndex> src_idx;
    std::vector<TensorIndex> dst_idx;
    src_idx.reserve(spatial_dim);
    dst_idx.reserve(spatial_dim);
    for (int dim = 0; dim < spatial_dim; ++dim)
    {
      const int shift = off[dim];
      const int64_t size = labels.size(dim);
      if (shift > 0)
      {
        src_idx.emplace_back(Slice(shift, size));
        dst_idx.emplace_back(Slice(0, size - shift));
      }
      else if (shift < 0)
      {
        src_idx.emplace_back(Slice(0, size + shift));
        dst_idx.emplace_back(Slice(-shift, size));
      }
      else
      {
        src_idx.emplace_back(Slice());
        dst_idx.emplace_back(Slice());
      }
    }

    auto src = labels.index(src_idx);
    auto dst = labels.index(dst_idx);
    auto mask = (src >= 0) & (dst >= 0) & (src != dst);
    if (!mask.any().item<bool>())
      continue;
    auto a = src.masked_select(mask);
    auto b = dst.masked_select(mask);
    edge_pairs.push_back(torch::stack({a, b}, 1));
  }

  if (edge_pairs.empty())
  {
    result.unique_labels =
        torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    result.adjacency =
        torch::zeros({0, 0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return result;
  }

  auto pairs = torch::cat(edge_pairs, 0).to(torch::kCPU).to(torch::kInt64);
  auto uniq = torch::_unique(pairs.view({-1}), /*sorted=*/true, /*return_inverse=*/false);
  result.unique_labels = std::get<0>(uniq);
  const int64_t n_lbl = result.unique_labels.numel();
  if (n_lbl == 0)
  {
    result.adjacency =
        torch::zeros({0, 0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return result;
  }

  const int64_t max_lbl = result.unique_labels.max().item<int64_t>();
  std::vector<int64_t> map(static_cast<size_t>(max_lbl + 1), -1);
  const auto * uniq_ptr = result.unique_labels.data_ptr<int64_t>();
  for (int64_t i = 0; i < n_lbl; ++i)
    map[uniq_ptr[i]] = i;

  result.adjacency =
      torch::zeros({n_lbl, n_lbl}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  const auto stride0 = static_cast<int64_t>(result.adjacency.stride(0));
  auto * adj_ptr = result.adjacency.data_ptr<int64_t>();

  const auto * pair_ptr = pairs.data_ptr<int64_t>();
  const int64_t n_pairs = pairs.size(0);
  for (int64_t i = 0; i < n_pairs; ++i)
  {
    const int64_t a = pair_ptr[2 * i];
    const int64_t b = pair_ptr[2 * i + 1];
    if (a == b || a < 0 || b < 0)
      continue;
    const int64_t ia = map[a];
    const int64_t ib = map[b];
    adj_ptr[ia * stride0 + ib] = 1;
    adj_ptr[ib * stride0 + ia] = 1;
  }

  return result;
}

torch::Tensor
buildOldColorTable(const std::vector<torch::Tensor> & per_color_labels,
                   const std::vector<int64_t> & offsets,
                   int n_colors)
{
  if (per_color_labels.size() != offsets.size())
    mooseError("per_color_labels and offsets must have the same length.");
  if (n_colors <= 0)
    mooseError("Number of colors must be positive.");

  int64_t max_label = -1;
  for (size_t c = 0; c < per_color_labels.size(); ++c)
  {
    auto lbl_cpu = per_color_labels[c].to(torch::kCPU).to(torch::kInt64);
    auto valid = lbl_cpu >= 0;
    if (valid.any().item<bool>())
    {
      const auto max_local = lbl_cpu.masked_select(valid).max().item<int64_t>();
      max_label = std::max(max_label, offsets[c] + max_local);
    }
  }
  if (max_label < 0)
    return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  std::vector<int64_t> table(static_cast<size_t>(max_label + 1), -1);
  for (size_t c = 0; c < per_color_labels.size(); ++c)
  {
    auto lbl_cpu = per_color_labels[c].to(torch::kCPU).to(torch::kInt64);
    auto uniq = torch::_unique(lbl_cpu.masked_select(lbl_cpu >= 0), /*sorted=*/true, /*return_inverse=*/false);
    auto uniq_vals = std::get<0>(uniq);
    const auto * ptr = uniq_vals.data_ptr<int64_t>();
    for (int64_t i = 0; i < uniq_vals.numel(); ++i)
      table[offsets[c] + ptr[i]] = static_cast<int64_t>(c);
  }

  return torch::from_blob(table.data(),
                          {static_cast<long>(table.size())},
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .clone();
}

torch::Tensor
buildNewColorTable(const torch::Tensor & unique_labels, const std::vector<unsigned int> & colors)
{
  if (!unique_labels.defined())
    mooseError("unique_labels tensor is undefined.");
  auto uniq_cpu = unique_labels.to(torch::kCPU).to(torch::kInt64);
  if (static_cast<size_t>(uniq_cpu.numel()) != colors.size())
    mooseError("unique_labels and colors size mismatch.");

  if (uniq_cpu.numel() == 0)
    return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  const int64_t max_lbl = uniq_cpu.max().item<int64_t>();
  std::vector<int64_t> table(static_cast<size_t>(max_lbl + 1), -1);
  const auto * ptr = uniq_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < uniq_cpu.numel(); ++i)
    table[ptr[i]] = static_cast<int64_t>(colors[i]);

  return torch::from_blob(table.data(),
                          {static_cast<long>(table.size())},
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .clone();
}

torch::Tensor
buildLabelColorGrid(const torch::Tensor & labels,
                    const torch::Tensor & unique_labels,
                    const std::vector<unsigned int> & colors)
{
  if (!labels.defined())
    mooseError("labels tensor is undefined.");

  auto color_table = buildNewColorTable(unique_labels, colors);
  auto grid_cpu = torch::full_like(labels.to(torch::kCPU).to(torch::kInt64), -1);
  if (color_table.numel() == 0)
    return grid_cpu.to(labels.device());

  auto flat = grid_cpu.view({-1});
  auto labels_cpu = labels.to(torch::kCPU).to(torch::kInt64).view({-1});
  auto mask = labels_cpu >= 0;
  if (mask.any().item<bool>())
  {
    auto idx = labels_cpu.masked_select(mask);
    auto mapped = color_table.index({idx});
    flat.index_put_({mask}, mapped);
  }

  return grid_cpu.view(labels.sizes()).to(labels.device());
}

std::vector<unsigned int>
colorAdjacencyWithPetsc(const torch::Tensor & adjacency,
                        unsigned int n_colors,
                        const std::string & algorithm)
{
  if (!adjacency.defined())
    mooseError("Adjacency tensor is undefined.");
  if (adjacency.dim() != 2 || adjacency.size(0) != adjacency.size(1))
    mooseError("Adjacency tensor must be square.");
  if (n_colors == 0)
    mooseError("Number of colors must be positive.");

  auto adj_cpu = adjacency.to(torch::kCPU).contiguous();
  const int64_t n = adj_cpu.size(0);
  std::vector<unsigned int> colors(static_cast<size_t>(n),
                                   std::numeric_limits<unsigned int>::max());
  if (n == 0)
    return colors;

  // Force double for PETSc; zero out diagonal.
  auto adj_double = adj_cpu.to(torch::kDouble);
  auto eye = torch::eye(n, adj_double.options());
  adj_double = torch::where(eye > 0, torch::zeros_like(adj_double), adj_double);

  // Copy to a contiguous std::vector<PetscScalar> (PETSc expects column-major dense,
  // but MatSeqDenseSetPreallocation copies the buffer; we follow the same layout used
  // by PolycrystalICTools::AdjacencyMatrix (row-major) which PETSc accepts for dense).
  const auto numel = static_cast<size_t>(adj_double.numel());
  std::vector<PetscScalar> dense(numel);
  std::memcpy(dense.data(), adj_double.data_ptr<double>(), numel * sizeof(double));

  Moose::PetscSupport::colorAdjacencyMatrix(
      dense.data(), static_cast<unsigned int>(n), n_colors, colors, algorithm.c_str());
  return colors;
}

std::vector<ComponentMeta>
computeComponentMetadata(const torch::Tensor & labels, int color, int halo_width, int rank)
{
  if (!labels.defined())
    mooseError("Labels tensor is undefined.");
  const int spatial_dim = static_cast<int>(labels.dim());
  if (spatial_dim != 2 && spatial_dim != 3)
    mooseError("Labels tensor must be 2D or 3D.");

  auto labels_i64 = labels.to(torch::kInt64).contiguous();
  auto valid = labels_i64 >= 0;
  if (!valid.any().item<bool>())
    return {};

  const int64_t max_label = labels_i64.max().item<int64_t>();
  const int64_t n_comp = max_label + 1;
  auto flat = labels_i64.reshape({-1});
  auto valid_flat = valid.reshape({-1});
  auto idx = flat.masked_select(valid_flat);

  auto float_opts = MooseTensor::floatTensorOptions().device(labels.device());

  auto ones = torch::ones(idx.sizes(), float_opts);
  auto volumes = torch::bincount(idx, ones, n_comp).to(float_opts.dtype());

  std::array<torch::Tensor, 3> sums{
      torch::zeros({n_comp}, float_opts),
      torch::zeros({n_comp}, float_opts),
      torch::zeros({n_comp}, float_opts)};

  if (spatial_dim == 2)
  {
    auto y = torch::arange(labels.size(0), float_opts);
    auto x = torch::arange(labels.size(1), float_opts);
    auto grids = torch::meshgrid({y, x}, "ij");
    auto y_flat = grids[0].reshape({-1}).masked_select(valid_flat);
    auto x_flat = grids[1].reshape({-1}).masked_select(valid_flat);
    sums[1].scatter_add_(0, idx, y_flat);
    sums[2].scatter_add_(0, idx, x_flat);
  }
  else
  {
    auto z = torch::arange(labels.size(0), float_opts);
    auto y = torch::arange(labels.size(1), float_opts);
    auto x = torch::arange(labels.size(2), float_opts);
    auto grids = torch::meshgrid({z, y, x}, "ij");
    auto z_flat = grids[0].reshape({-1}).masked_select(valid_flat);
    auto y_flat = grids[1].reshape({-1}).masked_select(valid_flat);
    auto x_flat = grids[2].reshape({-1}).masked_select(valid_flat);
    sums[0].scatter_add_(0, idx, z_flat);
    sums[1].scatter_add_(0, idx, y_flat);
    sums[2].scatter_add_(0, idx, x_flat);
  }

  auto vol_cpu = volumes.to(torch::kCPU);
  auto sum0 = sums[0].to(torch::kCPU);
  auto sum1 = sums[1].to(torch::kCPU);
  auto sum2 = sums[2].to(torch::kCPU);

  std::vector<ComponentMeta> meta(static_cast<size_t>(n_comp));
  for (int64_t i = 0; i < n_comp; ++i)
  {
    auto & m = meta[static_cast<size_t>(i)];
    m.rank = rank;
    m.color = color;
    m.local_label = i;
    m.volume = static_cast<int64_t>(vol_cpu[i].item<double>());
    if (m.volume > 0)
    {
      const double denom = static_cast<double>(m.volume);
      m.centroid[0] = sum0[i].item<double>() / denom;
      m.centroid[1] = sum1[i].item<double>() / denom;
      m.centroid[2] = sum2[i].item<double>() / denom;
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
  auto co = old_colors.to(gid_flat.device()).index({gid_rows});
  auto cn = new_colors.to(gid_flat.device()).index({gid_rows});
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

  auto new_colors = initial_colors;
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
