/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
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
TensorAveragePostprocessor::execute()
{
  _average = _u.sum().cpu().item<double>() / torch::numel(_u);
}

PostprocessorValue
TensorAveragePostprocessor::getValue() const
{
  return _average;
}
