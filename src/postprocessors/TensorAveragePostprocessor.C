/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorAveragePostprocessor.h"

registerMooseObject("MarlinApp", TensorAveragePostprocessor);

InputParameters
TensorAveragePostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the average value over a buffer.");
  return params;
}

TensorAveragePostprocessor::TensorAveragePostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
TensorAveragePostprocessor::initialize()
{
  _sum = 0.0;
  _numel = 0;
}

void
TensorAveragePostprocessor::execute()
{
  const auto owned = _buffer_base.ownedView();
  _sum = owned.sum().cpu().item<double>();
  _numel = torch::numel(owned);
}

void
TensorAveragePostprocessor::finalize()
{
  gatherSum(_sum);
  gatherSum(_numel);
  _average = _sum / _numel;
}

PostprocessorValue
TensorAveragePostprocessor::getValue() const
{
  return _average;
}
