/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObject.h"
#include "TensorProblem.h"

class TensorProblemInterface
{
public:
  TensorProblemInterface(MooseObject * moose_object)
    : _tensor_problem(TensorProblem::cast(
          moose_object,
          *moose_object->parameters().getCheckedPointerParam<SubProblem *>("_subproblem")))
  {
  }

protected:
  TensorProblem & _tensor_problem;
};
