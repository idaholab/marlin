/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBufferBase.h"
#include "DomainAction.h"
#include "TensorProblem.h"

InputParameters
TensorBufferBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("Generic TensorBuffer object.");
  params.registerBase("TensorBuffer");
  params.registerSystemAttributeName("TensorBuffer");
  params.addParam<bool>("reciprocal", false, "Is this a reciprocal space tensor?");
  params.addParam<std::vector<AuxVariableName>>(
      "map_to_aux_variable", {}, "Sync the given AuxVariable to the buffer contents");
  params.addParam<std::vector<AuxVariableName>>(
      "map_from_aux_variable", {}, "Sync the given AuxVariable to the buffer contents");

  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);

  return params;
}

TensorBufferBase::TensorBufferBase(const InputParameters & parameters)
  : MooseObject(parameters),
    DomainInterface(this),
    _reciprocal(getParam<bool>("reciprocal")),
    _domain_shape(getParam<bool>("reciprocal") ? _domain.getReciprocalShape() : _domain.getShape()),
    _options(_reciprocal ? MooseTensor::complexFloatTensorOptions()
                         : MooseTensor::floatTensorOptions())
{
  // const auto & map_to_aux_variable =
  // getParam<std::vector<AuxVariableName>>("map_to_aux_variable"); if (map_to_aux_variable.size() >
  // 1)
  //   paramError("mapping to multiple variables is not supported.");

  // if (!map_to_aux_variable.empty() && !_value_shape_buffer.empty())
  //   paramError("mapping non-scalar tensors is not supported.");

  // if (!getParam<std::vector<AuxVariableName>>("map_from_aux_variable").empty())
  //   paramError("functionality is not yet implemented.");
}

TensorBufferBase &
TensorBufferBase::operator=(const torch::Tensor & /*rhs*/)
{
  // TODO: remove
  //  if (this != &rhs)
  //  {
  //    torch::Tensor::operator=(rhs);
  //    expand();
  //  }
  return *this;
}

void
TensorBufferBase::expand()
{
  // try
  // {
  //   this->expand(_shape);
  // }
  // catch (const std::exception & e)
  // {
  //   mooseError("Assignment of incompatible data to tensor '", MooseBase::name(), "'");
  // }
}
