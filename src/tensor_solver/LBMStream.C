/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStream.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMStream);

namespace
{
/// write torch::roll(src, shift, {0, 1, 2}) into dst block by block, without temporaries
void
periodicShiftInto(torch::Tensor dst, const torch::Tensor & src, const std::array<int64_t, 3> & shift)
{
  // per dimension: (dst start, src start, length) of the one or two periodic segments
  std::array<std::vector<std::array<int64_t, 3>>, 3> segments;
  for (int d = 0; d < 3; ++d)
  {
    const auto n = src.size(d);
    const auto s = ((shift[d] % n) + n) % n;
    segments[d] = {{s, 0, n - s}};
    if (s != 0)
      segments[d].push_back({0, n - s, s});
  }

  for (const auto & a : segments[0])
    for (const auto & b : segments[1])
      for (const auto & c : segments[2])
        dst.narrow(0, a[0], a[2])
            .narrow(1, b[0], b[2])
            .narrow(2, c[0], c[2])
            .copy_(src.narrow(0, a[1], a[2]).narrow(1, b[1], b[2]).narrow(2, c[1], c[2]));
}
}

InputParameters
LBMStream::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription("LBM Streaming operation.");
  params.addParam<std::vector<TensorOutputBufferName>>(
      "buffer", {}, "The buffer this solver is writing to");

  params.addParam<std::vector<TensorInputBufferName>>("f_old", {}, "Old time step distribution");
  return params;
}

LBMStream::LBMStream(const InputParameters & parameters)
  : TensorSolver(parameters),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(_tensor_problem)),
    _stencil(_lb_problem.getStencil()),
    _radius(_lb_problem.getGhostRadius())
{
  std::vector<TensorOutputBufferName> output_buffer_names =
      getParam<std::vector<TensorOutputBufferName>>("buffer");

  std::vector<TensorInputBufferName> input_buffer_names =
      getParam<std::vector<TensorInputBufferName>>("f_old");

  const auto n = output_buffer_names.size();

  if (input_buffer_names.size() != n || output_buffer_names.size() != n)
    paramError("buffer", "Must have the same number of entries as 'f_old'");

  // ensure input and output buffers are distinct
  for (const auto i : make_range(n))
  {
    if (input_buffer_names[i] == output_buffer_names[i])
      paramError("f_old",
                 "Input buffer and output buffer must be distinct; buffer '",
                 input_buffer_names[i],
                 "' was provided for both.");
    _variables.push_back(Variable{getOutputBufferByName(output_buffer_names[i]),
                                  getBufferOldByName(input_buffer_names[i], 1, _radius)});
  }
}

void
LBMStream::computeBuffer()
{
  // Ensure all input buffers have the same ghost layers
  for (const auto & [name, ghost] : _input_buffer_ghost_layers)
    if (ghost != _radius)
      mooseError(
          "Ghost layer mismatch! Expected all input buffers to have the same ghost layer size'",
          _radius,
          "' but buffer '",
          name,
          "' has ghost layer size '",
          ghost,
          "'.");

  // Exchange ghost layers
  for (const auto & [buffer_name, ghost] : _input_buffer_ghost_layers)
    if (ghost > 0)
      _lb_problem.exchangeGhostLayers(buffer_name, ghost);

  // extract the lattice shifts once (item() is a device sync on GPU)
  if (_shifts.empty())
    for (int i = 0; i < _stencil._q; i++)
      _shifts.push_back({_stencil._ex[i].item<int64_t>(),
                         _stencil._ey[i].item<int64_t>(),
                         _stencil._ez[i].item<int64_t>()});

  const auto n_old = _variables[0]._f_old.size();
  if (n_old != 0 && _radius == 0)
  {
    // serial (periodic) streaming
    for (const auto v : index_range(_variables))
    {
      auto & [u, f_old] = _variables[v];
      // do not overwrite previous; every entry is written below, so nothing needs copying
      u = nextBuffer(v, u);
      for (int i = 0; i < _stencil._q; i++)
        periodicShiftInto(u.select(3, i), f_old[0].select(3, i), _shifts[i]);
      _lb_problem.maskedFillSolids(u, 0);
    }
  }
  else if (n_old != 0 && _radius > 0)
  {
    // streaming with ghost layers
    const auto owned = _lb_problem.getExtendedShape();
    const int64_t halo = _lb_problem.getGhostRadius();

    for (const auto v : index_range(_variables))
    {
      auto & [u, f_old] = _variables[v];
      // detach from old state (advanceState does a shallow copy, so u and
      // u_old share storage; copying into separate storage prevents in-place streaming from
      // corrupting the old state that boundary conditions may still need)
      u = nextBuffer(v, u).copy_(u);

      // define the owned region of u
      auto u_owned = u;
      for (unsigned int d = 0; d < _dim; d++)
        u_owned = u_owned.narrow(d, halo, owned[d]);

      for (int i = 0; i < _stencil._q; i++)
      {
        // stream via slicing with pre-extracted shifts
        auto f_shifted = f_old[0];
        for (unsigned int d = 0; d < _dim; d++)
          f_shifted = f_shifted.narrow(d, halo - _shifts[i][d], owned[d]);

        u_owned.select(3, i).copy_(f_shifted.select(3, i));
      }

      _lb_problem.maskedFillSolids(u_owned, 0);
    }
  }
}

torch::Tensor
LBMStream::nextBuffer(std::size_t v, const torch::Tensor & u)
{
  // Reuse the tensor this variable held two substeps ago when nothing else references its
  // storage anymore, instead of allocating (and page-faulting) a new lattice every substep.
  if (_spare.size() < _variables.size())
    _spare.resize(_variables.size());

  auto & spare = _spare[v];
  const bool reusable = spare.defined() && spare.use_count() == 1 &&
                        spare.storage().use_count() == 1 && spare.sizes() == u.sizes() &&
                        spare.strides() == u.strides() && spare.dtype() == u.dtype() &&
                        spare.device() == u.device();
  auto next = reusable ? spare : torch::empty_like(u);
  spare = u;
  return next;
}
