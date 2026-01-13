/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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
  params.addClassDescription("TensorSolver object.");
  params.registerBase("TensorSolver");

  params.addParam<TensorComputeName>(
      "root_compute",
      "Primary compute object that updates the buffers. This is usually a "
      "ComputeGroup object. A ComputeGroup encompassing all computes will be generated "
      "automatically if the user does not provide this parameter.");

  params.addParam<unsigned int>("substeps", 1, "Solver substeps per time step.");

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
  : TensorOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    // it is the derived solver's responsibility to update the _sub_dt and _sub_time (we can
    // probably come up with a better design here)
    // TODO: LBM doesn't need those :-/ - need a base class :-D
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime()),
    _dt(_tensor_problem.dt()),
    _dt_old(_tensor_problem.dtOld())
{
  const auto & forward_buffer_names = getParam<TensorOutputBufferName, TensorOutputBufferName>(
      "forward_buffer", "forward_buffer_new");
  for (const auto & [forward_buffer, forward_buffer_new] : forward_buffer_names)
    _forwarded_buffers.emplace_back(getOutputBufferByName(forward_buffer),
                                    getInputBufferByName(forward_buffer_new));
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOld(const std::string & param, unsigned int max_states, unsigned int ghost_layers)
{
  return getBufferOldByName(getParam<TensorInputBufferName>(param), max_states, ghost_layers);
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOldByName(const TensorInputBufferName & buffer_name, unsigned int max_states, unsigned int ghost_layers)
{   
  _input_buffer_ghost_layers[buffer_name] =
      std::max(_input_buffer_ghost_layers[buffer_name], ghost_layers);
  _tensor_problem.registerGhostLayerRequest(buffer_name, ghost_layers);
  return _tensor_problem.getBufferOld(buffer_name, max_states, ghost_layers);
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

void
TensorSolver::computeBuffer()
{
  _sub_time = _time;
  _sub_dt = _dt / _substeps;

  for (_substep = 0; _substep < _substeps; _substep++)
  {
    // perform the actual sub timestep
    substep();

    // we skip the advanceState on the last substep because MOOSE will call that automatically
    if (_substep < _substeps - 1)
      _tensor_problem.advanceState();

    // increment substep time
    _sub_time += _sub_dt;
  }
}
