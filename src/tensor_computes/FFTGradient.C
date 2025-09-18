/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTGradient.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", FFTGradient);

InputParameters
FFTGradient::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Tensor gradient.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>("input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  params.addRequiredParam<MooseEnum>(
      "direction", MooseEnum("X=0 Y=1 Z=2"), "Which axis to take the gradient along.");
  return params;
}

FFTGradient::FFTGradient(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _direction(getParam<MooseEnum>("direction")),
    _i(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
FFTGradient::computeBuffer()
{
  _u = _domain.ifft((_input_is_reciprocal ? _input : _domain.fft(_input)) *
                    _domain.getReciprocalAxis(_direction) * _i);
}
