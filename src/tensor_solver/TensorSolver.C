/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorSolver.h"
#include "TensorProblem.h"
#include "MarlinTypes.h"

InputParameters
TensorSolver::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.registerBase("TensorSolver");
  params.addParam<TensorComputeName>(
      "root_compute",
      "Primary compute object that updates the buffers. This is usually a "
      "ComputeGroup object. A ComputeGroup encompassing all computes will be generated "
      "automatically if the user does not provide this parameter.");
  params.addClassDescription("TensorSolver object.");

  params.addParam<std::vector<TensorOutputBufferName>>(
      "forward_buffer",
      {},
      "These buffers are updated with the corresponding buffers from forward_buffer_old. No "
      "integration is performed. Buffer forwarding is used only to resolve cyclic dependencies.");
  params.addParam<std::vector<TensorInputBufferName>>(
      "forward_buffer_new", {}, "New values to update `forward_buffer` with.");

  return params;
}

TensorSolver::TensorSolver(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _dt(_tensor_problem.dt()), _dt_old(_tensor_problem.dtOld())
{
  const auto & forward_buffer_names = getParam<TensorOutputBufferName, TensorOutputBufferName>(
      "forward_buffer", "forward_buffer_new");
  for (const auto & [forward_buffer, forward_buffer_new] : forward_buffer_names)
    _forwarded_buffers.emplace_back(getOutputBufferByName(forward_buffer),
                                    getInputBufferByName(forward_buffer_new));
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(getParam<TensorInputBufferName>(param), max_states);
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOldByName(const TensorInputBufferName & buffer_name, unsigned int max_states)
{
  return _tensor_problem.getBufferOld(buffer_name, max_states);
}

void
TensorSolver::updateDependencies()
{
  // the compute that's being solved for (usually a ComputeGroup)
  const auto & root_name = getParam<TensorComputeName>("root_compute");
  for (const auto & cmp : _tensor_problem.getComputes())
    if (cmp->name() == root_name)
    {
      _compute = cmp;
      _compute->updateDependencies();
      return;
    }

  paramError("root_compute", "Compute object not found.");
}

void
TensorSolver::forwardBuffers()
{
  for (const auto & [forward_buffer, forward_buffer_new] : _forwarded_buffers)
    forward_buffer = forward_buffer_new;
}
