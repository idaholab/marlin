/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTSemiImplicit.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", FFTSemiImplicit);

InputParameters
FFTSemiImplicit::validParams()
{
  InputParameters params = TensorTimeIntegrator::validParams();
  params.addClassDescription("Semi-implicit time integrator.");
  params.addRequiredParam<TensorInputBufferName>(
      "reciprocal_buffer", "Buffer with the reciprocal of the integrated buffer");
  params.addRequiredParam<TensorInputBufferName>(
      "linear_reciprocal", "Buffer with the reciprocal of the linear prefactor (e.g. kappa*k^2)");
  params.addRequiredParam<TensorInputBufferName>(
      "nonlinear_reciprocal", "Buffer with the reciprocal of the non-linear contribution");
  params.addParam<unsigned int>(
      "history_size", 1, "How many old states to use (determines time integration order).");
  return params;
}

FFTSemiImplicit::FFTSemiImplicit(const InputParameters & parameters)
  : TensorTimeIntegrator<>(parameters),
    _history_size(getParam<unsigned int>("history_size")),
    _reciprocal_buffer(getInputBuffer("reciprocal_buffer")),
    _linear_reciprocal(getInputBuffer("linear_reciprocal")),
    _non_linear_reciprocal(getInputBuffer("nonlinear_reciprocal")),
    _old_reciprocal_buffer(getBufferOld("reciprocal_buffer", _history_size)),
    _old_non_linear_reciprocal(getBufferOld("nonlinear_reciprocal", _history_size))
{
}

void
FFTSemiImplicit::computeBuffer()
{
  const auto n_old = std::min(_old_reciprocal_buffer.size(), _old_non_linear_reciprocal.size());

  torch::Tensor ubar;
  if (n_old == 0)
    // compute FFT time update (1st order)
    ubar = (_reciprocal_buffer + _sub_dt * _non_linear_reciprocal) / (1.0 - _sub_dt * _linear_reciprocal);

  if (n_old >= 1)
    // compute FFT time update (2nd order) - this probably breaks for adaptive dt!
    // ubar = (4.0 * _reciprocal_buffer - _old_reciprocal_buffer[0] +
    //         (2.0 * _sub_dt) * (2.0 * _non_linear_reciprocal - _old_non_linear_reciprocal[0])) /
    //        (3.0 - (2.0 * _sub_dt) * _linear_reciprocal);
    ubar = (_reciprocal_buffer +
            _sub_dt / 2.0 * (3.0 * _non_linear_reciprocal - _old_non_linear_reciprocal[0])) /
           (1.0 - _sub_dt * _linear_reciprocal);

  _u = _domain.ifft(ubar);
}
