/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddLBMBCAction.h"
#include "LatticeBoltzmannProblem.h"

registerMooseAction("MarlinApp", AddLBMBCAction, "add_tensor_bc");

InputParameters
AddLBMBCAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add LBM boundary condition object.");
  return params;
}

AddLBMBCAction::AddLBMBCAction(const InputParameters & parameters) : MooseObjectAction(parameters)
{
}

void
AddLBMBCAction::act()
{
  auto lb_problem = std::dynamic_pointer_cast<LatticeBoltzmannProblem>(_problem);
  if (!lb_problem)
    mooseError(
        "LBM BCs are only supported if the problem class is set to `LatticeBoltzmannProblem`");

  if (_current_task == "add_tensor_bc")
    lb_problem->addTensorBoundaryCondition(_type, _name, _moose_object_pars);
}
