/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddLBMStencilAction.h"
#include "LatticeBoltzmannProblem.h"

registerMooseAction("MarlinApp", AddLBMStencilAction, "add_stencil");

InputParameters
AddLBMStencilAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add LBM stencil object to the simulation.");
  return params;
}

AddLBMStencilAction::AddLBMStencilAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddLBMStencilAction::act()
{
  auto lb_problem = std::dynamic_pointer_cast<LatticeBoltzmannProblem>(_problem);
  if (!lb_problem)
    mooseError(
        "LBM Stencils are only supported if the problem class is set to `LatticeBoltzmannProblem`");

  if (_current_task == "add_stencil")
    lb_problem->addStencil(_type, _name, _moose_object_pars);
}
