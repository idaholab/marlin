/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorIntegralPostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", TensorIntegralPostprocessor);

InputParameters
TensorIntegralPostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorIntegralPostprocessor::TensorIntegralPostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
TensorIntegralPostprocessor::execute()
{
  const auto owned = _buffer_base.ownedView();
  _integral = owned.sum().cpu().item<double>();

  const auto s = _domain.getDomainMax() - _domain.getDomainMin();
  for (const auto dim : make_range(_domain.getDim()))
    _integral *= s(dim);

  _integral /= owned.numel();
}

void
TensorIntegralPostprocessor::finalize()
{
  gatherSum(_integral);
}

PostprocessorValue
TensorIntegralPostprocessor::getValue() const
{
  return _integral;
}
