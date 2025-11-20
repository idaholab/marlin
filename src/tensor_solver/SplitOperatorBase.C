/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SplitOperatorBase.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
SplitOperatorBase::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription("Base class for non-linear/linear operator splits.");

  params.addRequiredParam<std::vector<TensorOutputBufferName>>(
      "buffer", "The buffer this solver is writing to");

  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "reciprocal_buffer", "Buffer with the reciprocal of the integrated buffer");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "linear_reciprocal",
      "Buffer with the reciprocal of the linear prefactor (e.g. kappa*k^2). Either one buffer per "
      "nonlinear_reciprocal, or no buffer names, or `0` to skip linear reciprocal buffers for a "
      "given variable.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "nonlinear_reciprocal", "Buffer with the reciprocal of the non-linear contribution");
  return params;
}

SplitOperatorBase::SplitOperatorBase(const InputParameters & parameters) : TensorSolver(parameters)
{
}

void
SplitOperatorBase::getVariables(unsigned int history_size)
{
  auto buffers = getParam<std::vector<TensorOutputBufferName>>("buffer");
  auto reciprocal_buffers = getParam<std::vector<TensorInputBufferName>>("reciprocal_buffer");
  auto linear_reciprocals = getParam<std::vector<TensorInputBufferName>>("linear_reciprocal");
  auto nonlinear_reciprocals = getParam<std::vector<TensorInputBufferName>>("nonlinear_reciprocal");

  const auto n = buffers.size();

  if (linear_reciprocals.empty())
    linear_reciprocals.assign(n, "0");

  if (reciprocal_buffers.size() != n || linear_reciprocals.size() != n ||
      nonlinear_reciprocals.size() != n)
    paramError("buffer",
               "Must have the same number of entries as 'reciprocal_buffer', 'linear_reciprocal' "
               "and 'nonlinear_reciprocal'.");

  for (const auto i : make_range(n))
    _variables.push_back(Variable{
        getOutputBufferByName(buffers[i]),
        getInputBufferByName(reciprocal_buffers[i]),
        linear_reciprocals[i] == "0" ? nullptr : &getInputBufferByName(linear_reciprocals[i]),
        getInputBufferByName(nonlinear_reciprocals[i]),
        getBufferOldByName(nonlinear_reciprocals[i], history_size)});
}
