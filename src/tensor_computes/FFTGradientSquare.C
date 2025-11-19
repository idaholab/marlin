/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTGradientSquare.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", FFTGradientSquare);

InputParameters
FFTGradientSquare::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Tensor gradient.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  params.addParam<Real>("factor", 1.0, "Prefactor to the gradient square");
  return params;
}

FFTGradientSquare::FFTGradientSquare(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _factor(getParam<Real>("factor")),
    _dim(_domain.getDim()),
    _i(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
FFTGradientSquare::computeBuffer()
{
  auto sqr = [](const auto & a) { return a * a; };

  const auto r = _input_is_reciprocal ? _input : _domain.fft(_input);

  _u = sqr(_domain.ifft(r * _domain.getReciprocalAxis(0) * _i));
  for (const auto d : make_range(1u, _dim))
    _u += sqr(_domain.ifft(r * _domain.getReciprocalAxis(d) * _i));

  if (_factor != 1.0)
    _u *= _factor;
}
