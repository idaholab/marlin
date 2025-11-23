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
  if (_max_ghost_layers == 0)
  {
    _u = torch::zeros(_domain.getValueShape(getParam<std::vector<int64_t>>("value_dimensions")),
                      MooseTensor::floatTensorOptions());
  }
  else
  {
    auto shape = _domain.getValueShape(getParam<std::vector<int64_t>>("value_dimensions"));
    // Add padding to spatial dimensions (last 3 dims)
    // Assuming domain dims are the last ones.
    // DomainAction::getValueShape returns (extra..., z, y, x) or similar.
    // We need to pad the spatial dimensions.
    const unsigned int dim = _domain.getDim();
    for (unsigned int i = 0; i < dim; ++i)
      shape[shape.size() - 1 - i] += 2 * _max_ghost_layers;

    _padded_u = torch::zeros(shape, MooseTensor::floatTensorOptions());

    // Create slice for _u (center)
    std::vector<torch::indexing::TensorIndex> slice;
    for (size_t i = 0; i < shape.size() - dim; ++i)
      slice.push_back(torch::indexing::Slice()); // All extra dims
    for (unsigned int i = 0; i < dim; ++i)
      slice.push_back(torch::indexing::Slice(_max_ghost_layers, -_max_ghost_layers));

    _u = _padded_u.index(slice);

    _u = _padded_u.index(slice);

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
        pair.second = _u;
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

    // Ensure view 0 is in the map if not already (though getTensor(0) handles it by returning _u
    // directly, but if someone called getTensor(0) and it created an entry...) Actually
    // getTensor(0) returns _u member, not from map. But if we want consistency, we can put it in
    // map.
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
