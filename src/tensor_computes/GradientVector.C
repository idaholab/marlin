/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GradientVector.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", GradientVector);

InputParameters
GradientVector::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Vector gradient of a scalar tensor.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  return params;
}

GradientVector::GradientVector(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _zero(torch::tensor(0.0, MooseTensor::floatTensorOptions()))
{
}

void
GradientVector::computeBuffer()
{
  if (_u.dim() < static_cast<int64_t>(_dim) + 1)
    mooseError("GradientVector output buffer must have value_dimensions = '3'.");
  if (_u.size(_dim) != 3)
    mooseError("GradientVector output buffer must have value_dimensions = '3'.");

  auto r = _input_is_reciprocal ? _input : _domain.fft(_input);
  auto i_recip = r * _imaginary;

  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;

  auto component_index = [&](int comp)
  {
    std::vector<TensorIndex> idx;
    idx.reserve(_u.dim());
    for (unsigned int d = 0; d < _dim; ++d)
      idx.emplace_back(Slice());
    idx.emplace_back(comp);
    for (int64_t d = static_cast<int64_t>(_dim) + 1; d < _u.dim(); ++d)
      idx.emplace_back(Slice());
    return idx;
  };

  if (_dim >= 1)
    _u.index_put_(component_index(0), _domain.ifft(i_recip * _i));
  if (_dim >= 2)
    _u.index_put_(component_index(1), _domain.ifft(i_recip * _j));
  else
    _u.index_put_(component_index(1), _zero);

  if (_dim >= 3)
    _u.index_put_(component_index(2), _domain.ifft(i_recip * _k));
  else
    _u.index_put_(component_index(2), _zero);
}
