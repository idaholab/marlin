/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorExtremeValuePostprocessor.h"

registerMooseObject("MarlinApp", TensorExtremeValuePostprocessor);

InputParameters
TensorExtremeValuePostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Find extreme values in the Tensor buffer");
  MooseEnum valueType("MIN MAX");
  params.addParam<MooseEnum>("value_type", valueType, "Extreme value type");
  return params;
}

TensorExtremeValuePostprocessor::TensorExtremeValuePostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters),
    _value_type(getParam<MooseEnum>("value_type").getEnum<ValueType>())
{
}

void
TensorExtremeValuePostprocessor::execute()
{
  _value = _value_type == ValueType::MIN ? torch::min(_u).cpu().item<double>()
                                         : torch::max(_u).cpu().item<double>();
}

PostprocessorValue
TensorExtremeValuePostprocessor::getValue() const
{
  return _value;
}
