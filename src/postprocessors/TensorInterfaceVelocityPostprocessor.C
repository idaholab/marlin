/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorInterfaceVelocityPostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", TensorInterfaceVelocityPostprocessor);

InputParameters
TensorInterfaceVelocityPostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  params.addParam<Real>("gradient_threshold",
                        1e-3,
                        "Ignore cells with a gradient component magnitude below this threshold.");
  return params;
}

TensorInterfaceVelocityPostprocessor::TensorInterfaceVelocityPostprocessor(
    const InputParameters & parameters)
  : TensorPostprocessor(parameters),
    _u_old(_tensor_problem.getBufferOld(getParam<TensorInputBufferName>("buffer"), 1)),
    _dim(_domain.getDim()),
    _i(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _gradient_threshold(getParam<Real>("gradient_threshold"))
{
}

void
TensorInterfaceVelocityPostprocessor::execute()
{
  if (_u_old.empty())
  {
    _velocity = 0.0;
    return;
  }

  const auto du = (_u - _u_old[0]) / _dt; // TODO: _dt_old?
  torch::Tensor vsquare;
  for (const auto i : make_range(_dim))
  {
    const auto grad = _domain.ifft(_domain.fft(_u) * _domain.getReciprocalAxis(i) * _i);
    const auto v = torch::where(torch::abs(grad) > 1e-3, du / grad, 0.0);
    if (i == 0)
      vsquare = v * v;
    else
      vsquare += v * v;
  }

  _velocity = std::sqrt(torch::max(vsquare).item<double>());
}

PostprocessorValue
TensorInterfaceVelocityPostprocessor::getValue() const
{
  return _velocity;
}
