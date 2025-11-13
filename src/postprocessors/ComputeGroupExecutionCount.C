
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeGroupExecutionCount.h"
#include "ComputeGroup.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", ComputeGroupExecutionCount);

InputParameters
ComputeGroupExecutionCount::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription(
      "Return the number of computeBuffer() calls issued to the given compute group object.");
  params.addParam<TensorComputeName>(
      "compute_group", "root", "ComputeGroup TensorCompute object to get execution count from.");
  return params;
}

ComputeGroupExecutionCount::ComputeGroupExecutionCount(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _tensor_problem(TensorProblem::cast(this, this->_fe_problem)),
    _compute_group(
        _tensor_problem.getCompute<ComputeGroup>(getParam<TensorComputeName>("compute_group")))
{
}

PostprocessorValue
ComputeGroupExecutionCount::getValue() const
{
  return _compute_group.getComputeCount();
}
