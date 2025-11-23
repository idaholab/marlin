/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#include "PlainTensorBuffer.h"
#include "MarlinUtils.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", PlainTensorBuffer);

InputParameters
PlainTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addParam<std::vector<int64_t>>("value_dimensions", {}, "Optional value dimensions");
  return params;
}

PlainTensorBuffer::PlainTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters)
{
}

void
PlainTensorBuffer::init()
{
  auto shape = _domain.getValueShape(getParam<std::vector<int64_t>>("value_dimensions"));
  const unsigned int dim = _domain.getDim();

  if (_max_ghost_layers == 0)
  {
    _u = torch::zeros(shape, MooseTensor::floatTensorOptions());
    _padded_u = _u;
    _unpadded_u = _u;
  }
  else
  {
    // Add padding to spatial dimensions (last 3 dims)
    auto padded_shape = shape;
    for (unsigned int i = 0; i < dim; ++i)
      padded_shape[padded_shape.size() - 1 - i] += 2 * _max_ghost_layers;

    _padded_u = torch::zeros(padded_shape, MooseTensor::floatTensorOptions());

    // _u is now the full padded tensor (default for computes)
    _u = _padded_u;

    // Create unpadded view for output
    std::vector<torch::indexing::TensorIndex> slice;
    for (size_t i = 0; i < shape.size() - dim; ++i)
      slice.push_back(torch::indexing::Slice()); // All extra dims
    for (unsigned int i = 0; i < dim; ++i)
      slice.push_back(torch::indexing::Slice(
          static_cast<int64_t>(_max_ghost_layers), -static_cast<int64_t>(_max_ghost_layers)));

    _unpadded_u = _padded_u.index(slice);
  }

  // Initialize views map
  // We need to update all views that have been requested
  for (auto & pair : _views)
  {
    unsigned int ghosts = pair.first;
    if (ghosts == _max_ghost_layers)
    {
      pair.second = _padded_u;
    }
    else if (ghosts == 0)
    {
      pair.second = _unpadded_u;
    }
    else
    {
      // Slice _padded_u to get 'ghosts' layers
      // The padding is _max_ghost_layers.
      // We want center + ghosts.
      // Start index: _max_ghost_layers - ghosts
      // End index: -(_max_ghost_layers - ghosts)

      std::vector<torch::indexing::TensorIndex> view_slice;
      unsigned int diff = _max_ghost_layers - ghosts;
      for (size_t i = 0; i < shape.size() - dim; ++i)
        view_slice.push_back(torch::indexing::Slice());
      for (unsigned int i = 0; i < dim; ++i)
        view_slice.push_back(torch::indexing::Slice(diff, -diff));

      pair.second = _padded_u.index(view_slice);
    }
  }
}

void
PlainTensorBuffer::makeCPUCopy()
{
  if (!_u.defined())
    return;

  if (_cpu_copy_requested)
  {
    if (_u.is_cpu())
      _u_cpu = _u.clone().contiguous();
    else
      _u_cpu = _u.cpu().contiguous();
  }
}
