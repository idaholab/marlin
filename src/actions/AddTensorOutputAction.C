/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddTensorOutputAction.h"
#include "TensorProblem.h"

registerMooseAction("MarlinApp", AddTensorOutputAction, "add_tensor_output");

InputParameters
AddTensorOutputAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an TensorOutput object to the simulation.");
  return params;
}

AddTensorOutputAction::AddTensorOutputAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorOutputAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor objects are only supported if the problem class is set to `TensorProblem`");

  tensor_problem->addTensorOutput(_type, _name, _moose_object_pars);
}
