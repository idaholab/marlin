/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/**********************************************************************/

#include "HaloCommunication.h"

#include "DomainAction.h"
#include "MooseError.h"

#include <mpi.h>
#include <vector>

namespace HaloCommunication
{

void
exchangeGhostTensor(torch::Tensor & tensor, unsigned int ghost_layers, const DomainAction & domain)
{
  if (ghost_layers == 0)
    return;

  const unsigned int dim = domain.getDim();
  const auto owned = domain.getLocalGridSize();
  const auto global = domain.getGridSize();
  const auto periodic = domain.getPeriodicDirections();

  for (unsigned int d = 0; d < dim; ++d)
    if (ghost_layers > static_cast<unsigned int>(owned[d]))
      mooseError("Requested ghost layers (",
                 ghost_layers,
                 ") exceed owned extent (",
                 owned[d],
                 ") in direction ",
                 d,
                 ".");

  const auto sizes = tensor.sizes();
  if (sizes.size() < dim)
    mooseError("Tensor rank is smaller than spatial dimension for ghost exchange.");

  int64_t halo = -1;
  for (unsigned int d = 0; d < dim; ++d)
  {
    const int64_t h = (sizes[d] - owned[d]) / 2;
    if (halo < 0)
      halo = h;
    else if (halo != h)
      mooseError("Non-uniform halo widths are not supported for ghost exchange.");
  }

  if (halo < static_cast<int64_t>(ghost_layers))
    mooseError(
        "Requested ghost layers (", ghost_layers, ") exceed allocated halo width (", halo, ").");

  const bool use_gpu = domain.gpuAwareMPI() && tensor.is_cuda();
  const auto device_options = tensor.options();
  const auto cpu_options = tensor.options().device(torch::kCPU);
  const auto mpi_type = domain.getMPIType(tensor.scalar_type());

  int my_rank = 0, n_ranks = 1;
  MPI_Comm_rank(domain.getMPIComm(), &my_rank);
  MPI_Comm_size(domain.getMPIComm(), &n_ranks);

  // The decomposition is derived generically from the per-rank owned boxes, so
  // this works for any axis-aligned box decomposition: REAL_SPACE grids, FFT
  // slabs (one split dimension), and FFT pencils (two split dimensions).
  std::array<int64_t, 3> my_begin, my_end;
  domain.getLocalBounds(my_rank, my_begin, my_end);

  // find the rank owning the box that matches my bounds in all dimensions
  // except d, where its owned range begins (or ends) at the given target
  const auto findNeighbor = [&](unsigned int d, int64_t target, bool match_begin) -> int
  {
    for (int r = 0; r < n_ranks; ++r)
    {
      std::array<int64_t, 3> begin, end;
      domain.getLocalBounds(r, begin, end);
      bool match = (match_begin ? begin[d] : end[d]) == target;
      for (unsigned int e = 0; e < dim; ++e)
        if (e != d && (begin[e] != my_begin[e] || end[e] != my_end[e]))
          match = false;
      if (match)
        return r;
    }
    mooseError("No neighbor rank found along direction ", d, " (target ", target, ").");
  };

  for (unsigned int d = 0; d < dim; ++d)
  {
    // unpartitioned dimension: fill the halo by periodic wrap if applicable
    if (owned[d] == global[d])
    {
      if (periodic[d])
      {
        const auto owned_width = owned[d];
        auto wrap_upper = tensor.narrow(d, halo, ghost_layers);
        auto wrap_lower = tensor.narrow(d, halo + owned_width - ghost_layers, ghost_layers);
        tensor.narrow(d, halo - ghost_layers, ghost_layers).copy_(wrap_lower);
        tensor.narrow(d, halo + owned_width, ghost_layers).copy_(wrap_upper);
      }
      continue;
    }

    const bool has_lower = my_begin[d] > 0;
    const bool has_upper = my_end[d] < global[d];

    struct NeighborExchange
    {
      int neighbor_rank;
      torch::Tensor send_buf;
      torch::Tensor recv_buf;
      torch::Tensor recv_view;
      int send_tag;
      int recv_tag;
    };

    std::vector<NeighborExchange> exchanges;
    exchanges.reserve(2);

    auto prepare_neighbor = [&](bool lower)
    {
      int neighbor_rank;
      if (lower)
      {
        if (!has_lower && !periodic[d])
          return;
        // lower neighbor: the rank whose owned range ends at my begin (or at
        // the global extent for the periodic wrap)
        neighbor_rank = findNeighbor(d, has_lower ? my_begin[d] : global[d], /*match_begin=*/false);
      }
      else
      {
        if (!has_upper && !periodic[d])
          return;
        // upper neighbor: the rank whose owned range begins at my end (or 0 for wrap)
        neighbor_rank = findNeighbor(d, has_upper ? my_end[d] : 0, /*match_begin=*/true);
      }

      const int64_t send_start =
          lower ? halo : halo + owned[d] - static_cast<int64_t>(ghost_layers);
      const int64_t recv_start =
          lower ? halo - static_cast<int64_t>(ghost_layers) : halo + owned[d];

      auto send_slice = tensor.narrow(d, send_start, ghost_layers).contiguous();
      auto recv_view = tensor.narrow(d, recv_start, ghost_layers);

      if (neighbor_rank == my_rank)
      {
        recv_view.copy_(send_slice);
        return;
      }

      auto send_buf = use_gpu ? send_slice : send_slice.to(cpu_options);
      auto recv_buf = torch::empty_like(send_buf, use_gpu ? device_options : cpu_options);

      const int send_tag = 200 + d * 2 + (lower ? 0 : 1);
      const int recv_tag = 200 + d * 2 + (lower ? 1 : 0);

      exchanges.push_back({neighbor_rank, send_buf, recv_buf, recv_view, send_tag, recv_tag});
    };

    prepare_neighbor(true);
    prepare_neighbor(false);

    if (!exchanges.empty())
    {
      std::vector<MPI_Request> reqs(2 * exchanges.size(), MPI_REQUEST_NULL);
      std::size_t idx = 0;
      for (const auto & ex : exchanges)
        MPI_Irecv(ex.recv_buf.data_ptr(),
                  ex.recv_buf.numel(),
                  mpi_type,
                  ex.neighbor_rank,
                  ex.recv_tag,
                  domain.getMPIComm(),
                  &reqs[idx++]);
      for (const auto & ex : exchanges)
        MPI_Isend(ex.send_buf.data_ptr(),
                  ex.send_buf.numel(),
                  mpi_type,
                  ex.neighbor_rank,
                  ex.send_tag,
                  domain.getMPIComm(),
                  &reqs[idx++]);

      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

      for (auto & ex : exchanges)
      {
        if (!use_gpu)
          ex.recv_buf = ex.recv_buf.to(device_options);
        ex.recv_view.copy_(ex.recv_buf);
      }
    }
  }
}

} // namespace HaloCommunication
