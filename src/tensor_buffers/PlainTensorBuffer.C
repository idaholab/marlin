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
    _unpadded_slice.clear();
  }
  else
  {
    // Add padding to spatial dimensions (last 3 dims)
    auto padded_shape = shape;
    for (unsigned int i = 0; i < dim; ++i)
      padded_shape[padded_shape.size() - 1 - i] += 2 * _max_ghost_layers;

    _u = torch::zeros(padded_shape, MooseTensor::floatTensorOptions());

    // Create slice for unpadded view
    _unpadded_slice.clear();
    for (size_t i = 0; i < shape.size() - dim; ++i)
      _unpadded_slice.push_back(torch::indexing::Slice()); // All extra dims
    for (unsigned int i = 0; i < dim; ++i)
      _unpadded_slice.push_back(torch::indexing::Slice(
          static_cast<int64_t>(_max_ghost_layers), -static_cast<int64_t>(_max_ghost_layers)));
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
