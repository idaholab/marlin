/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AddTensorComputeAction.h"
#include "TensorProblem.h"
#include "hit/parse.h"
#include "LatticeBoltzmannProblem.h"

registerMooseAction("MarlinApp", AddTensorComputeAction, "add_tensor_ic");
registerMooseAction("MarlinApp", AddTensorComputeAction, "add_tensor_bc");
registerMooseAction("MarlinApp", AddTensorComputeAction, "add_tensor_compute");
registerMooseAction("MarlinApp", AddTensorComputeAction, "add_tensor_postprocessor");

InputParameters
AddTensorComputeAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();

  params.set<std::string>("type") = "";
  params.addParam<bool>("skip_param_construction", true, "Allow for empty type default.");
  params.suppressParameter<bool>("skip_param_construction");

  params.addClassDescription("Add an TensorOperator object to the simulation.");
  return params;
}

AddTensorComputeAction::AddTensorComputeAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
  if (_type == "")
    _type = "ComputeGroup";
  _moose_object_pars = _factory.getValidParams(_type);
  _moose_object_pars.blockFullpath() = parameters.blockFullpath();
}

void
AddTensorComputeAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  auto lb_poblem = std::dynamic_pointer_cast<LatticeBoltzmannProblem>(_problem);

  if (!tensor_problem)
    mooseError("Tensor objects are only supported if the problem class is set to `TensorProblem`");

  // use addObject<Tensorxxxxxx>(_type, _name, _moose_object_pars, /* threaded = */ false) ?

  // automatically populate `computes` parameter with subblocks
  if (_type == "ComputeGroup")
  {
    auto & computes = _moose_object_pars.set<std::vector<TensorComputeName>>("computes");
    std::set<TensorComputeName> computes_set(computes.begin(), computes.end());

    // Node::children should be marked const. using this as a temporary workaround.
    auto * node = const_cast<hit::Node *>(_moose_object_pars.getHitNode());
    const auto children = node->children(hit::NodeType::Section);
    for (const auto child : children)
      computes_set.insert(child->path());

    computes.clear();
    std::copy(computes_set.begin(), computes_set.end(), std::back_inserter(computes));
  }

  if (_current_task == "add_tensor_ic")
    tensor_problem->addTensorComputeInitialize(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_compute")
    tensor_problem->addTensorComputeSolve(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_postprocessor")
    tensor_problem->addTensorComputePostprocess(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_bc")
    lb_poblem->addTensorBoundaryCondition(_type, _name, _moose_object_pars);
}
