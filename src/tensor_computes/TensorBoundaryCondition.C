/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBoundaryCondition.h"
#include "DomainAction.h"

InputParameters
TensorBoundaryCondition::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addRequiredParam<TensorInputBufferName>("variable",
                                                 "The field this boundary condition applies to.");
  MooseEnum boundary("left right bottom top front back");
  params.addRequiredParam<MooseEnum>(
      "boundary", boundary, "Which face of the domain this boundary condition applies to.");
  params.addClassDescription("Base class for boundary conditions on a non-periodic real-space "
                             "domain edge.");
  return params;
}

TensorBoundaryCondition::TensorBoundaryCondition(const InputParameters & parameters)
  : TensorOperatorBase(parameters),
    _face(getParam<MooseEnum>("boundary").getEnum<Face>()),
    _variable_name(getParam<TensorInputBufferName>("variable")),
    _owns_face(false)
{
  const auto rank = _domain.comm().rank();
  std::array<int64_t, 3> begin, end;
  _domain.getLocalBounds(rank, begin, end);
  const auto & n_global = _domain.getGridSize();

  const auto d = faceDim();
  _owns_face = isUpperFace() ? (end[d] == n_global[d]) : (begin[d] == 0);
}
