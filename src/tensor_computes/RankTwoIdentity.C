/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RankTwoIdentity.h"
#include "MarlinUtils.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", RankTwoIdentity);

InputParameters
RankTwoIdentity::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Rank two identity tensor in real space.");
  return params;
}

RankTwoIdentity::RankTwoIdentity(const InputParameters & parameters)
  : TensorOperator(parameters), _dim(_domain.getDim())
{
}

void
RankTwoIdentity::computeBuffer()
{
  const auto full = _domain.getValueShape({_dim, _dim});
  _u = torch::eye(_dim, MooseTensor::floatTensorOptions()).expand(full);
}
