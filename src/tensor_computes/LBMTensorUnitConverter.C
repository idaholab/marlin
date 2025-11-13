/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMTensorUnitConverter.h"

#if 0

registerMooseObject("MarlinApp", LBMTensorUnitConverter);

InputParameters
LBMTensorUnitConverter::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("tensor_buffer",
                                                 "The macroscopic buffer variable to convert");
  params.addRequiredParam<std::vector<std::string>>("constant",
                                                    "The scalar conversion constant names");
  params.addClassDescription("LBMConstantTensor object.");
  return params;
}

LBMTensorUnitConverter::LBMTensorUnitConverter(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _tensor_buffer(getInputBuffer(getParam<TensorInputBufferName>("tensor_buffer"))),
    _conversion_constant(_lb_problem.getConstant<Real>(getParam<TensorInputBufferName>("constant")))
{
mooseWarning("Unit converter is under development and not tested.")
}

void
LBMTensorUnitConverter::computeBuffer()
{
  _u = _tensor_buffer * _conversion_constant;
}

#endif
