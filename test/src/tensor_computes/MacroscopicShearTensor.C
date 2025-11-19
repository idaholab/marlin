/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MacroscopicShearTensor.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", MacroscopicShearTensor);

InputParameters
MacroscopicShearTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("DeGeus mechanics test material.");
  params.addParam<TensorInputBufferName>("F", "F", "Deformation gradient tensor.");
  return params;
}

MacroscopicShearTensor::MacroscopicShearTensor(const InputParameters & parameters)
  : TensorOperator<>(parameters), _tF(getInputBuffer<>("F"))
{
}

void
MacroscopicShearTensor::computeBuffer()
{
  const auto avg = _domain.average(_tF);

  // macroscopic loading
  auto applied_shear = torch::eye(_dim, MooseTensor::floatTensorOptions());
  applied_shear.index_put_({torch::indexing::Ellipsis, 0, 1},
                           applied_shear.index({torch::indexing::Ellipsis, 0, 1}) + _time);

  _u = (applied_shear - avg);
}
