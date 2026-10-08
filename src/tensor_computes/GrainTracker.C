/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainTracker.h"

#include "DomainAction.h"
#include "HaloCommunication.h"
#include "MarlinUtils.h"
#include "TensorBuffer.h"
#include "TensorProblem.h"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <numeric>

registerMooseObject("MarlinApp", GrainTracker);

namespace
{

/// Allgather a variable-length int64 vector from all ranks (identity for one rank).
std::vector<int64_t>
allgatherInt64(const std::vector<int64_t> & local, MPI_Comm comm, int n_ranks)
{
  if (n_ranks == 1)
    return local;

  const int n_local = static_cast<int>(local.size());
  std::vector<int> counts(n_ranks);
  MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displs(n_ranks, 0);
  for (int r = 1; r < n_ranks; ++r)
    displs[r] = displs[r - 1] + counts[r - 1];
  const int total = displs[n_ranks - 1] + counts[n_ranks - 1];

  std::vector<int64_t> result(total);
  MPI_Allgatherv(local.data(),
                 n_local,
                 MPI_INT64_T,
                 result.data(),
                 counts.data(),
                 displs.data(),
                 MPI_INT64_T,
                 comm);
  return result;
}

std::vector<std::pair<int64_t, int64_t>>
allgatherPairs(const std::vector<std::pair<int64_t, int64_t>> & local, MPI_Comm comm, int n_ranks)
{
  std::vector<int64_t> flat;
  flat.reserve(local.size() * 2);
  for (const auto & [a, b] : local)
  {
    flat.push_back(a);
    flat.push_back(b);
  }
  const auto all = allgatherInt64(flat, comm, n_ranks);

  std::vector<std::pair<int64_t, int64_t>> pairs;
  pairs.reserve(all.size() / 2);
  for (std::size_t i = 0; i + 1 < all.size(); i += 2)
    pairs.emplace_back(all[i], all[i + 1]);
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  return pairs;
}

} // namespace

InputParameters
GrainTracker::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addClassDescription(
      "Tracks grains in a multi-order-parameter phase field model, maintains persistent grain "
      "identities, and remaps grains to different order parameters when grains sharing an order "
      "parameter approach each other closer than the exclusion distance.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "op_buffers", "Order parameter buffers (one per order parameter, remapped in place).");
  params.addParam<Real>("threshold", 0.1, "Phase field threshold for grain detection.");
  params.addRangeCheckedParam<unsigned int>(
      "halo_width",
      2,
      "halo_width >= 1",
      "Exclusion distance in cells. Grains sharing an order parameter are remapped when their "
      "footprints dilated by this distance touch; remapping moves the field within the same "
      "dilated footprint, so choose a few interface widths.");
  MooseEnum connectivity("minimal full", "minimal");
  params.addParam<MooseEnum>(
      "connectivity",
      connectivity,
      "Neighbor connectivity for grain detection: minimal (face neighbors, 4/6) or full "
      "(including corners, 8/26).");
  params.addParam<Real>("tracking_tolerance",
                        3.0,
                        "Maximum centroid travel distance (in cells) between tracking "
                        "steps for matching grains.");
  params.addParam<Real>("tracking_volume_ratio",
                        4.0,
                        "Maximum relative volume change between tracking steps for matching "
                        "grains (0 disables the check).");
  params.addRangeCheckedParam<unsigned int>(
      "interval", 1, "interval > 0", "Run the tracker every `interval` executions.");
  params.addParam<TensorOutputBufferName>(
      "grain_id_buffer", "Optional output buffer holding the persistent grain id per cell.");
  params.addParam<bool>(
      "remap_old_states", true, "Also remap old states (time integrator history).");
  MooseEnum on_conflict("error warn", "error");
  params.addParam<MooseEnum>("on_conflict",
                             on_conflict,
                             "Behavior when grains cannot be assigned conflict-free order "
                             "parameters (more mutually close grains than order parameters).");
  return params;
}

GrainTracker::GrainTracker(const InputParameters & parameters)
  : TensorOperatorBase(parameters),
    _op_names(getParam<std::vector<TensorInputBufferName>>("op_buffers")),
    _threshold(getParam<Real>("threshold")),
    _halo_width(getParam<unsigned int>("halo_width")),
    _connectivity(getParam<MooseEnum>("connectivity")),
    _tracking_tolerance(getParam<Real>("tracking_tolerance")),
    _tracking_volume_ratio(getParam<Real>("tracking_volume_ratio")),
    _interval(getParam<unsigned int>("interval")),
    _remap_old_states(getParam<bool>("remap_old_states")),
    _on_conflict(getParam<MooseEnum>("on_conflict")),
    _grain_id_buffer(nullptr),
    _next_persistent_id(0),
    _execution_count(0),
    _total_remaps(0),
    _last_conflicts(0)
{
  if (_op_names.empty())
    paramError("op_buffers", "At least one order parameter buffer is required.");

  const unsigned int ghost = _domain.isRealSpaceMode() ? _halo_width : 0;
  for (const auto & name : _op_names)
  {
    getInputBufferByName(name, ghost);
    _op_tensors.push_back(&_tensor_problem.getBuffer<torch::Tensor>(name));
    _op_buffers.push_back(dynamic_cast<TensorBuffer<torch::Tensor> *>(&getBufferBase(name)));
  }

  if (isParamValid("grain_id_buffer"))
    _grain_id_buffer =
        &getOutputBufferByName<torch::Tensor>(getParam<TensorOutputBufferName>("grain_id_buffer"));
}

void
GrainTracker::check()
{
  const auto owned = _domain.getLocalGridSize();
  const auto global = _domain.getGridSize();

  bool partitioned = false;
  for (unsigned int d = 0; d < _dim; ++d)
    if (owned[d] != global[d])
      partitioned = true;

  if (_domain.isRealSpaceMode() || partitioned)
    for (unsigned int d = 0; d < _dim; ++d)
      if (static_cast<int64_t>(_halo_width) > owned[d])
        mooseError("halo_width (",
                   _halo_width,
                   ") exceeds the owned local extent (",
                   owned[d],
                   ") in direction ",
                   d,
                   ".");
}

void
GrainTracker::computeBuffer()
{
  if (_execution_count++ % _interval != 0)
    return;
  trackAndRemap();
}

std::vector<torch::Tensor>
GrainTracker::cropOpBuffers(int64_t buffer_pad, int64_t hw)
{
  const auto owned = _domain.getLocalGridSize();
  std::vector<torch::Tensor> crops;
  crops.reserve(_op_tensors.size());
  for (std::size_t c = 0; c < _op_tensors.size(); ++c)
  {
    auto & buf = *_op_tensors[c];
    if (!buf.defined())
      mooseError("Order parameter buffer '", _op_names[c], "' is not initialized.");
    // materialize lazy views (e.g. zero-stride expanded constants) so the remap
    // can write back into the buffer
    if (!buf.is_contiguous())
      buf = buf.contiguous();
    if (buf.dim() < static_cast<int64_t>(_dim))
      mooseError("Order parameter buffer '",
                 _op_names[c],
                 "' has fewer dimensions (",
                 buf.dim(),
                 ") than the domain (",
                 _dim,
                 "). Make sure its initial condition produces a full field.");
    for (unsigned int d = 0; d < _dim; ++d)
      if (buf.size(d) != owned[d] + 2 * buffer_pad)
        mooseError("Order parameter buffer '",
                   _op_names[c],
                   "' has unexpected extent ",
                   buf.size(d),
                   " in direction ",
                   d,
                   " (expected ",
                   owned[d] + 2 * buffer_pad,
                   ").");

    auto crop = buf;
    for (unsigned int d = 0; d < _dim; ++d)
      crop = crop.narrow(d, buffer_pad - hw, owned[d] + 2 * hw);
    crops.push_back(crop);
  }
  return crops;
}

void
GrainTracker::trackAndRemap()
{
  using namespace GrainRemap;

  const bool realspace = _domain.isRealSpaceMode();
  const auto comm = _domain.getMPIComm();
  int n_ranks = 1, rank = 0;
  MPI_Comm_size(comm, &n_ranks);
  MPI_Comm_rank(comm, &rank);

  const int spatial_dim = static_cast<int>(_dim);
  const int n_colors = static_cast<int>(_op_names.size());
  const auto owned = _domain.getLocalGridSize();
  const auto global = _domain.getGridSize();
  std::array<int64_t, 3> begin, end;
  _domain.getLocalBounds(rank, begin, end);
  const auto periodic = _domain.getPeriodicDirections();

  // Layouts:
  // - REAL_SPACE: buffers carry a permanent halo padding; ghosts are exchanged
  //   by the problem and the tracker works on cropped views (owned + hw ring).
  // - partitioned spectral (FFT_SLAB/FFT_PENCIL): buffers are unpadded owned
  //   slabs/pencils; the tracker assembles its own padded work tensor and fills
  //   the ring with an explicit ghost exchange.
  // - serial or replicated: full domain on every rank; periodicity is handled
  //   by in-tensor wrap, no ring is needed.
  bool partitioned = false;
  for (int d = 0; d < spatial_dim; ++d)
    if (owned[d] != global[d])
      partitioned = true;
  const bool spectral_parallel = !realspace && partitioned;

  const int64_t buffer_pad = realspace ? _tensor_problem.getMaxGhostLayer() : 0;
  const int64_t hw = realspace ? std::min<int64_t>(_halo_width, buffer_pad)
                               : (spectral_parallel ? _halo_width : 0);

  // ranks participating in tracking collectives; on replicated layouts every
  // rank sees the full domain and computes the identical result locally
  const bool distributed = realspace || partitioned;
  const int comm_ranks = distributed ? n_ranks : 1;
  const int my_part = distributed ? rank : 0;
  const bool ghosted = distributed && hw > 0;

  Geometry geom;
  geom.spatial_dim = spatial_dim;
  geom.pad = hw;
  for (int d = 0; d < spatial_dim; ++d)
  {
    geom.owned[d] = owned[d];
    geom.global[d] = global[d];
    geom.global_begin[d] = begin[d];
    geom.periodic[d] = periodic[d];
  }

  // without a ghost ring periodicity is handled by in-tensor wrap; with one it
  // is handled through the (wrap-copied or exchanged) halo ring
  std::array<bool, 3> wrap{{false, false, false}};
  if (!ghosted)
    for (int d = 0; d < spatial_dim; ++d)
      wrap[d] = periodic[d];

  GrainRemapOptions options;
  options.threshold = _threshold;
  options.connectivity =
      _connectivity == "full" ? (spatial_dim == 2 ? 8 : 26) : (spatial_dim == 2 ? 4 : 6);
  options.halo_width = static_cast<int>(_halo_width);
  options.tracking_tolerance = _tracking_tolerance;
  options.tracking_volume_ratio = _tracking_volume_ratio;
  options.wrap = wrap;

  // assemble the stacked working field (owned region + active halo ring)
  std::vector<torch::Tensor> crops;
  torch::Tensor eta;
  if (spectral_parallel)
  {
    // unpadded buffers: build a padded work tensor and fill the ring explicitly
    crops = cropOpBuffers(/*buffer_pad=*/0, /*hw=*/0);
    std::vector<int64_t> padded_shape;
    for (int d = 0; d < spatial_dim; ++d)
      padded_shape.push_back(owned[d] + 2 * hw);
    padded_shape.push_back(n_colors);
    eta = torch::zeros(padded_shape, crops[0].options());
    auto inner = eta;
    for (int d = 0; d < spatial_dim; ++d)
      inner = inner.narrow(d, hw, owned[d]);
    inner.copy_(torch::stack(crops, -1));
    HaloCommunication::exchangeGhostTensor(eta, hw, _domain);
  }
  else
  {
    crops = cropOpBuffers(buffer_pad, hw);
    eta = torch::stack(crops, -1);
  }

  // detection masks; blank out halo rings that no exchange refreshes
  auto masks = computeColorMasks(eta, _threshold);
  if (hw > 0)
    for (int d = 0; d < spatial_dim; ++d)
      if (owned[d] == global[d] && !periodic[d])
        for (auto & mask : masks)
        {
          mask.narrow(d, 0, hw).fill_(false);
          mask.narrow(d, hw + owned[d], hw).fill_(false);
        }

  // per-color connected component labeling and rank-local numbering
  std::vector<torch::Tensor> per_color_labels;
  per_color_labels.reserve(masks.size());
  for (const auto & mask : masks)
    per_color_labels.push_back(labelConnectedComponents(mask, options.connectivity, wrap));

  std::vector<int64_t> color_offsets, color_counts;
  auto labels = buildGlobalContiguousLabels(per_color_labels, color_offsets, color_counts);

  // globally unique label numbering across ranks
  std::vector<int64_t> all_counts(static_cast<std::size_t>(n_colors) * comm_ranks);
  if (comm_ranks == 1)
    all_counts = color_counts;
  else
    MPI_Allgather(
        color_counts.data(), n_colors, MPI_INT64_T, all_counts.data(), n_colors, MPI_INT64_T, comm);

  std::vector<int64_t> rank_base(comm_ranks, 0);
  int64_t n_labels = 0;
  std::vector<int> label_color;
  for (int r = 0; r < comm_ranks; ++r)
  {
    rank_base[r] = n_labels;
    for (int c = 0; c < n_colors; ++c)
    {
      const auto count = all_counts[static_cast<std::size_t>(r) * n_colors + c];
      label_color.insert(label_color.end(), count, c);
      n_labels += count;
    }
  }

  if (n_labels == 0)
  {
    // no grains anywhere; reset state (all ranks reach this together)
    _grains.clear();
    _last_conflicts = 0;
    if (_grain_id_buffer)
      *_grain_id_buffer =
          torch::full(_op_tensors[0]->sizes(), -1.0, MooseTensor::floatTensorOptions());
    return;
  }

  if (rank_base[my_part] != 0)
    labels = torch::where(labels >= 0, labels + rank_base[my_part], labels);

  // additive moments for all global labels (other ranks' entries stay zero)
  auto moments = computeComponentMoments(labels, n_labels, geom);
  if (comm_ranks > 1)
  {
    std::vector<double> packed(static_cast<std::size_t>(n_labels) * ComponentMoments::packed_size);
    for (int64_t l = 0; l < n_labels; ++l)
      moments[l].pack(packed.data() + l * ComponentMoments::packed_size);
    MPI_Allreduce(
        MPI_IN_PLACE, packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE, MPI_SUM, comm);
    for (int64_t l = 0; l < n_labels; ++l)
      moments[l].unpack(packed.data() + l * ComponentMoments::packed_size);
  }

  // stitch labels across rank seams and periodic boundaries via ghost exchange
  std::vector<std::pair<int64_t, int64_t>> equivalences;
  if (ghosted)
  {
    auto pre = labels.clone();
    HaloCommunication::exchangeGhostTensor(labels, hw, _domain);
    equivalences = detectSeamEquivalences(pre, labels, geom);
  }
  if (comm_ranks > 1)
    equivalences = allgatherPairs(equivalences, comm, comm_ranks);

  int64_t n_grains = 0;
  const auto label_to_grain = mergeLabels(n_labels, equivalences, n_grains);
  auto grains = finalizeGrains(moments, label_to_grain, label_color, n_grains, geom);

  // persistent grain identities
  matchPersistentGrains(_grains, grains, options, geom, _next_persistent_id);
  for (const auto & g : grains)
    _next_persistent_id = std::max(_next_persistent_id, g.persistent_id + 1);

  // adjacency between grains within 2*halo_width of each other
  auto grain_ids = applyLabelMap(labels, label_to_grain);
  auto expanded = expandLabels(grain_ids, static_cast<int>(_halo_width), wrap);
  if (ghosted)
    HaloCommunication::exchangeGhostTensor(expanded, hw, _domain);

  auto adjacency_pairs = extractAdjacencyPairs(expanded, options.connectivity, wrap);
  if (comm_ranks > 1)
    adjacency_pairs = allgatherPairs(adjacency_pairs, comm, comm_ranks);
  const auto adjacency = buildAdjacencyLists(n_grains, adjacency_pairs);

  // recolor, processing large grains first (deterministic across ranks)
  std::vector<int> initial_colors(grains.size());
  for (std::size_t i = 0; i < grains.size(); ++i)
    initial_colors[i] = grains[i].color;
  std::vector<int64_t> order(grains.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(),
            order.end(),
            [&grains](int64_t a, int64_t b)
            {
              if (grains[a].volume != grains[b].volume)
                return grains[a].volume > grains[b].volume;
              return a < b;
            });

  std::vector<int64_t> conflicts;
  const auto new_colors = greedyRecolor(adjacency, initial_colors, n_colors, order, conflicts);
  _last_conflicts = conflicts.size();
  if (!conflicts.empty())
  {
    std::string ids;
    for (const auto gid : conflicts)
      ids += " " + std::to_string(grains[gid].persistent_id);
    const auto message = "GrainTracker could not assign conflict-free order parameters to " +
                         std::to_string(conflicts.size()) + " grain(s) (persistent ids:" + ids +
                         "). Increase the number of order parameters or reduce halo_width.";
    if (_on_conflict == "error")
      mooseError(message);
    else
      mooseWarning(message);
  }

  std::vector<int> old_color_vec(grains.size());
  std::vector<int> new_color_vec(grains.size());
  std::vector<int64_t> remapped;
  for (std::size_t i = 0; i < grains.size(); ++i)
  {
    grains[i].new_color = new_colors[i];
    old_color_vec[i] = grains[i].color;
    new_color_vec[i] = new_colors[i];
    if (new_colors[i] != grains[i].color)
      remapped.push_back(static_cast<int64_t>(i));
  }

  // move recolored grains to their new order parameters (dilated footprint only)
  if (!remapped.empty())
  {
    std::vector<int64_t> changed_map(static_cast<std::size_t>(n_grains), -1);
    for (const auto gid : remapped)
      changed_map[gid] = gid;
    const auto changed_expanded = applyLabelMap(expanded, changed_map);

    remapOrderParameters(eta, changed_expanded, old_color_vec, new_color_vec);
    if (spectral_parallel)
    {
      // write back the owned region of the padded work tensor
      auto inner = eta;
      for (int d = 0; d < spatial_dim; ++d)
        inner = inner.narrow(d, hw, owned[d]);
      for (int c = 0; c < n_colors; ++c)
        crops[c].copy_(inner.select(-1, c));
    }
    else
      for (int c = 0; c < n_colors; ++c)
        crops[c].copy_(eta.select(-1, c));

    // remap the time integrator history consistently. Stored states have the
    // buffer layout: padded in REAL_SPACE mode, unpadded owned otherwise.
    if (_remap_old_states)
    {
      auto changed_states = changed_expanded;
      if (spectral_parallel)
        for (int d = 0; d < spatial_dim; ++d)
          changed_states = changed_states.narrow(d, hw, owned[d]);
      const int64_t state_off = realspace ? buffer_pad - hw : 0;
      const int64_t state_ring = realspace ? hw : 0;

      std::size_t n_states = std::numeric_limits<std::size_t>::max();
      for (auto * buf : _op_buffers)
        n_states = std::min(n_states, buf ? buf->getOldTensorRef().size() : std::size_t(0));
      if (n_states == std::numeric_limits<std::size_t>::max())
        n_states = 0;

      for (std::size_t s = 0; s < n_states; ++s)
      {
        std::vector<torch::Tensor> old_crops;
        bool usable = true;
        for (std::size_t c = 0; c < _op_buffers.size(); ++c)
        {
          auto & state = _op_buffers[c]->getOldTensorRef()[s];
          if (!state.defined() || !state.sizes().equals(_op_tensors[c]->sizes()))
          {
            usable = false;
            break;
          }
          auto crop = state;
          for (unsigned int d = 0; d < _dim; ++d)
            crop = crop.narrow(d, state_off, owned[d] + 2 * state_ring);
          old_crops.push_back(crop);
        }
        if (!usable)
          continue;

        auto old_eta = torch::stack(old_crops, -1);
        remapOrderParameters(old_eta, changed_states, old_color_vec, new_color_vec);
        for (int c = 0; c < n_colors; ++c)
          old_crops[c].copy_(old_eta.select(-1, c));
      }
    }

    _total_remaps += remapped.size();
  }

  // optional persistent grain id output for visualization/postprocessing
  if (_grain_id_buffer)
  {
    std::vector<int64_t> persistent_map(grains.size());
    for (std::size_t i = 0; i < grains.size(); ++i)
      persistent_map[i] = grains[i].persistent_id;
    auto id_grid = applyLabelMap(grain_ids, persistent_map);
    if (spectral_parallel)
      for (int d = 0; d < spatial_dim; ++d)
        id_grid = id_grid.narrow(d, hw, owned[d]);

    auto out = torch::full(_op_tensors[0]->sizes(), -1.0, MooseTensor::floatTensorOptions());
    auto out_crop = out;
    if (realspace)
      for (unsigned int d = 0; d < _dim; ++d)
        out_crop = out_crop.narrow(d, buffer_pad - hw, owned[d] + 2 * hw);
    out_crop.copy_(id_grid.to(out.dtype()));
    *_grain_id_buffer = out;
  }

  _console << "GrainTracker '" << name() << "': " << grains.size() << " grain(s)";
  if (!remapped.empty())
    _console << ", remapped " << remapped.size() << " grain(s)";
  _console << std::endl;

  _grains = std::move(grains);
}
