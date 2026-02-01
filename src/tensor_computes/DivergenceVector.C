/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "DivergenceVector.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", DivergenceVector);

InputParameters
DivergenceVector::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Divergence of a vector tensor.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<Real>("factor", 1.0, "Optional prefactor to apply to the divergence.");
  return params;
}

DivergenceVector::DivergenceVector(const InputParameters & parameters)
  : TensorOperator<>(parameters), _input(getInputBuffer("input")), _factor(getParam<Real>("factor"))
{
}

void
DivergenceVector::computeBuffer()
{
  if (_input.dim() < static_cast<int64_t>(_dim) + 1)
    mooseError("DivergenceVector input buffer must have value_dimensions = '3'.");
  if (_input.size(_dim) != 3)
    mooseError("DivergenceVector input buffer must have value_dimensions = '3'.");

  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;

  auto component_index = [&](int comp)
  {
    std::vector<TensorIndex> idx;
    idx.reserve(_input.dim());
    for (unsigned int d = 0; d < _dim; ++d)
      idx.emplace_back(Slice());
    idx.emplace_back(comp);
    for (int64_t d = static_cast<int64_t>(_dim) + 1; d < _input.dim(); ++d)
      idx.emplace_back(Slice());
    return idx;
  };

  torch::Tensor div;

  if (_dim >= 1)
  {
    const auto vx = _input.index(component_index(0));
    div = _domain.ifft(_domain.fft(vx) * _i * _imaginary);
  }

  if (_dim >= 2)
  {
    const auto vy = _input.index(component_index(1));
    auto term = _domain.ifft(_domain.fft(vy) * _j * _imaginary);
    div = div.defined() ? div + term : term;
  }

  if (_dim >= 3)
  {
    const auto vz = _input.index(component_index(2));
    auto term = _domain.ifft(_domain.fft(vz) * _k * _imaginary);
    div = div.defined() ? div + term : term;
  }

  if (!div.defined())
    div = torch::zeros_like(_u);

  _u = _factor * div;
}
