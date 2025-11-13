/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddTensorPredictorAction.h"
#include "TensorProblem.h"
#include "IterativeTensorSolverInterface.h"

registerMooseAction("MarlinApp", AddTensorPredictorAction, "add_tensor_predictor");

InputParameters
AddTensorPredictorAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an TensorPredictor object to the simulation.");
  return params;
}

AddTensorPredictorAction::AddTensorPredictorAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorPredictorAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor objects are only supported if the problem class is set to `TensorProblem`");

  if (_current_task != "add_tensor_predictor")
    return;

  // get the current solver
  auto solver = tensor_problem->getSolver<IterativeTensorSolverInterface>();

  // solver.addPredictor();
}
