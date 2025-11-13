/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStream.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMStream);

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
    _stencil(_lb_problem.getStencil())
{
  std::vector<TensorOutputBufferName> output_buffer_names =
      getParam<std::vector<TensorOutputBufferName>>("buffer");

  std::vector<TensorInputBufferName> input_buffer_names =
      getParam<std::vector<TensorInputBufferName>>("f_old");

  const auto n = output_buffer_names.size();

  if (input_buffer_names.size() != n || output_buffer_names.size() != n)
    paramError("buffer", "Must have the same number of entries as 'f_old'");

  for (const auto i : make_range(n))
    _variables.push_back(Variable{getOutputBufferByName(output_buffer_names[i]),
                                  getBufferOldByName(input_buffer_names[i], 1)});
}

void
LBMStream::computeBuffer()
{
  const auto n_old = _variables[0]._f_old.size();
  if (n_old != 0)
  {
    for (auto & [u, f_old] : _variables)
    {
      // do not overwrite previous
      u = u.clone();
      for (int i = 0; i < _stencil._q; i++)
      {
        u.index_put_({Slice(), Slice(), Slice(), i},
                     torch::roll(f_old[0].index({Slice(), Slice(), Slice(), i}),
                                 /* shifts = */
                                 {_stencil._ex[i].item<int64_t>(),
                                  _stencil._ey[i].item<int64_t>(),
                                  _stencil._ez[i].item<int64_t>()},
                                 /* dims = */
                                 {0, 1, 2}));
      }
      _lb_problem.maskedFillSolids(u, 0);
    }
  }
}
