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
                         : MooseTensor::floatTensorOptions()),
    _tensor_problem(getParam<TensorProblem *>("_tensor_problem"))
{
  if (!_tensor_problem)
    mooseError("TensorProblem pointer not provided to TensorBufferBase.");

  if (_domain.isRealSpaceMode() && _reciprocal)
    mooseError("Reciprocal space tensors are not supported in REAL_SPACE parallel mode.");
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

torch::Tensor
TensorBufferBase::ownedView() const
{
  const auto ghost = _tensor_problem->getMaxGhostLayer();
  auto t = getRawTensor();
  if (ghost == 0)
    return t;

  const unsigned int dim = _domain.getDim();
  if (t.dim() < static_cast<int64_t>(dim))
    mooseError("Owned view requested on tensor '", name(), "' with insufficient dimensions.");

  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;
  std::vector<TensorIndex> slices;
  slices.reserve(t.dim());
  const auto & owned = _domain.getLocalGridSize();
  for (unsigned int d = 0; d < dim; ++d)
    slices.emplace_back(Slice(ghost, ghost + owned[d]));
  for (int64_t d = dim; d < t.dim(); ++d)
    slices.emplace_back(Slice());

  return t.index(c10::ArrayRef<TensorIndex>(slices));
}
