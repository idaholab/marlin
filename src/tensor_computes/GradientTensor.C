/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#include "GradientTensor.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", GradientTensor);

InputParameters
GradientTensor::validParams()
{
  InputParameters params = TensorOperator<GradientTensorType>::validParams();
#ifdef NEML2_ENABLED
  params.addClassDescription("Gradient of the coupled tensor buffer.");
#else
  params.addClassDescription("Object requires NEML2.");
#endif

  params.addParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  return params;
}

GradientTensor::GradientTensor(const InputParameters & parameters)
  : TensorOperator<GradientTensorType>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _zero(torch::tensor(0.0, MooseTensor::floatTensorOptions()))
{
#ifndef NEML2_ENABLED
  mooseError("Object requires NEML2");
#endif
}

void
GradientTensor::computeBuffer()
{
#ifdef NEML2_ENABLED
  auto i_reciprocal_input = (_input_is_reciprocal ? _input : _domain.fft(_input)) * _imaginary;
  auto grad_x = neml2::Scalar(_domain.ifft(i_reciprocal_input * _i), _dim);
  auto grad_y = neml2::Scalar(_dim > 1 ? _domain.ifft(i_reciprocal_input * _j) : _zero, _dim);
  auto grad_z = neml2::Scalar(_dim > 2 ? _domain.ifft(i_reciprocal_input * _k) : _zero, _dim);
  _u = neml2::Vec::fill(grad_x, grad_y, grad_z);
#endif
}
