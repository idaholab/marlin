/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainTrackerPostprocessor.h"
#include "GrainTracker.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", GrainTrackerPostprocessor);

InputParameters
GrainTrackerPostprocessor::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription(
      "Report the grain count, cumulative remap count, or conflict count from a GrainTracker.");
  params.addRequiredParam<TensorComputeName>("grain_tracker", "GrainTracker compute object.");
  MooseEnum value_type("count remapped conflicts", "count");
  params.addParam<MooseEnum>("value_type", value_type, "Quantity to report.");
  return params;
}

GrainTrackerPostprocessor::GrainTrackerPostprocessor(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _tensor_problem(TensorProblem::cast(this, this->_fe_problem)),
    _grain_tracker(
        _tensor_problem.getCompute<GrainTracker>(getParam<TensorComputeName>("grain_tracker"))),
    _value_type(getParam<MooseEnum>("value_type"))
{
}

PostprocessorValue
GrainTrackerPostprocessor::getValue() const
{
  if (_value_type == "remapped")
    return _grain_tracker.getRemapCount();
  if (_value_type == "conflicts")
    return _grain_tracker.getConflictCount();
  return _grain_tracker.getGrainCount();
}
