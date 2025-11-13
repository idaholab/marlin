//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorIntegralChangePostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", TensorIntegralChangePostprocessor);

InputParameters
TensorIntegralChangePostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorIntegralChangePostprocessor::TensorIntegralChangePostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters), _u_old(_tensor_problem.getBufferOld(getParam<TensorInputBufferName>("buffer"), 1))
{
}

void
TensorIntegralChangePostprocessor::execute()
{
  {
  if (!_u_old.empty())
    _integral = torch::abs(_u - _u_old[0]).sum().cpu().item<double>();
  else
    _integral = torch::abs(_u).sum().cpu().item<double>();

  for (const auto dim : make_range(_domain.getDim()))
    _integral *= _domain.getGridSpacing()(dim);
  }
}

PostprocessorValue
TensorIntegralChangePostprocessor::getValue() const
{
  return _integral;
}
