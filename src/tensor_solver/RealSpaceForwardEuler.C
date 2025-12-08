/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RealSpaceForwardEuler.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", RealSpaceForwardEuler);

InputParameters
RealSpaceForwardEuler::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription("Real space forward Euler solver.");

  params.addParam<std::vector<TensorOutputBufferName>>(
      "buffer", {}, "The buffer this solver is writing to");
  params.addParam<std::vector<TensorInputBufferName>>(
      "time_derivative", {}, "Time derivative for the corresponding buffer");

  return params;
}

RealSpaceForwardEuler::RealSpaceForwardEuler(const InputParameters & parameters)
  : TensorSolver(parameters)
{
  auto buffers = getParam<std::vector<TensorOutputBufferName>>("buffer");
  auto time_derivatives = getParam<std::vector<TensorInputBufferName>>("time_derivative");

  const auto n = buffers.size();
  if (time_derivatives.size() != n)
    paramError("buffer", "Must have the same number of entries as 'time_derivatives'.");

  for (const auto i : make_range(n))
    _variables.push_back(Variable{
        getOutputBufferByName(buffers[i]),
        getInputBufferByName(time_derivatives[i]),
    });
}

void
RealSpaceForwardEuler::substep()
{
  // re-evaluate the solve compute
  _compute->realSpaceComputeBuffer();
  forwardBuffers();

  // integrate all variables
  for (auto & [u, time_derivative] : _variables)
    u = u + _sub_dt * time_derivative;
}
