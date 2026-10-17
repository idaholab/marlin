/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddTensorBCAction.h"
#include "TensorProblem.h"

registerMooseAction("MarlinApp", AddTensorBCAction, "add_tensor_boundary_condition");

InputParameters
AddTensorBCAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add a TensorBoundaryCondition object to the simulation.");
  return params;
}

AddTensorBCAction::AddTensorBCAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorBCAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor boundary conditions are only supported if the problem class is set to "
               "`TensorProblem` (or a subclass).");

  if (_current_task == "add_tensor_boundary_condition")
    tensor_problem->addTensorComputeBC(_type, _name, _moose_object_pars);
}
