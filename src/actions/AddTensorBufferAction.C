/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddTensorBufferAction.h"
#include "TensorProblem.h"

registerMooseAction("MarlinApp", AddTensorBufferAction, "add_tensor_buffer");

InputParameters
AddTensorBufferAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add a TensorBuffer object to the simulation.");
  params.set<std::string>("type") = "PlainTensorBuffer";
  return params;
}

AddTensorBufferAction::AddTensorBufferAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorBufferAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor Buffers are only supported if the problem class is set to `TensorProblem`");

  tensor_problem->addTensorBuffer(_type, _name, _moose_object_pars);
}
