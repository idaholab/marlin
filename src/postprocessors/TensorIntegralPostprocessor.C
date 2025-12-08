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
  : TensorAveragePostprocessor(parameters)
{
}

void
TensorIntegralPostprocessor::finalize()
{
  TensorAveragePostprocessor::finalize();
  const auto volume = RealVectorValue(1,1,1) * (_domain.getDomainMax() - _domain.getDomainMin());
  _integral = _average * volume;
}

PostprocessorValue
TensorIntegralPostprocessor::getValue() const
{
  return _integral;
}
