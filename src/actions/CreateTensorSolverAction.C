
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "CreateTensorSolverAction.h"
#include "TensorProblem.h"
#include "TensorSolver.h"
#include "MarlinTypes.h"

registerMooseAction("MarlinApp", CreateTensorSolverAction, "create_tensor_solver");

InputParameters
CreateTensorSolverAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Create a TensorSolver.");
  return params;
}

CreateTensorSolverAction::CreateTensorSolverAction(const InputParameters & parameters)
  : MooseObjectAction(parameters), DomainInterface(this)
{
}

void
CreateTensorSolverAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("A TensorSolver is only supported if the problem class is set to `TensorProblem`");

  // Add a pointer to the TensorProblem and the Domain
  _moose_object_pars.addPrivateParam<TensorProblem *>("_tensor_problem", tensor_problem.get());
  _moose_object_pars.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // check if a root compute was supplied, otherwise automatically generate one
  if (!_moose_object_pars.isParamValid("root_compute"))
  {
    mooseInfo("Automatically generating root compute.");

    // get the names of all computes
    std::vector<TensorComputeName> compute_names;
    for (const auto & cmp : tensor_problem->getComputes())
      compute_names.push_back(cmp->name());

    // create ComputeGroup
    const auto root_name = "automatic_root_compute";
    auto params = _factory.getValidParams("ComputeGroup");
    params.set<std::vector<TensorComputeName>>("computes") = compute_names;
    tensor_problem->addTensorComputeSolve("ComputeGroup", root_name, params);

    // set solver root compute to the generated object
    _moose_object_pars.set<TensorComputeName>("root_compute") = root_name;
  }

  // Create the object
  auto solver = _factory.create<TensorSolver>(_type, "TensorSolver", _moose_object_pars, 0);
  tensor_problem->setSolver(solver, {});
}
