/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "CoefficientTimeDerivativeOperator.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", CoefficientTimeDerivativeOperator);

InputParameters
CoefficientTimeDerivativeOperator::validParams()
{
  InputParameters params = TimeDerivativeOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("coefficient",
                                                 "The cell-centred coefficient buffer a >= 0");
  params.addClassDescription("Time derivative term a du/dt with a cell-centred coefficient a for "
                             "implicit solves, backward Euler or BDF2.");
  return params;
}

CoefficientTimeDerivativeOperator::CoefficientTimeDerivativeOperator(
    const InputParameters & parameters)
  : TimeDerivativeOperator(parameters),
    _coefficient_name(getParam<TensorInputBufferName>("coefficient")),
    _coefficient_base(getBufferBase(_coefficient_name))
{
}

torch::Tensor
CoefficientTimeDerivativeOperator::ownedCoefficient() const
{
  return _coefficient_base.ownedView();
}

torch::Tensor
CoefficientTimeDerivativeOperator::checkedCoefficient() const
{
  const auto a = ownedCoefficient();
  if (_domain.allreduceSum((a < 0).sum()).cpu().item<int64_t>() > 0)
    mooseError("The coefficient '",
               _coefficient_name,
               "' of CoefficientTimeDerivativeOperator '",
               name(),
               "' is negative in some cells; it must be nonnegative.");
  return a;
}

torch::Tensor
CoefficientTimeDerivativeOperator::rhsContribution() const
{
  // the coefficient is that of the time level of the unknown when the step starts; the action
  // and the stencil use the same a (a change within the step is an error, see computeBuffer)
  _step_coefficient = checkedCoefficient().clone();
  return _step_coefficient * TimeDerivativeOperator::rhsContribution();
}

void
CoefficientTimeDerivativeOperator::computeBuffer()
{
  // a nonlinear iteration reruns the computes writing a; the time derivative of a moving a is
  // not defined here, so it must keep the value the right-hand side was formed with
  if (!_step_coefficient.defined())
    mooseError("CoefficientTimeDerivativeOperator '",
               name(),
               "' was applied before the right-hand side of the step was formed.");
  if (!torch::equal(ownedCoefficient(), _step_coefficient))
    mooseError("The coefficient '",
               _coefficient_name,
               "' of CoefficientTimeDerivativeOperator '",
               name(),
               "' changed within a time step, which is not supported.");

  TimeDerivativeOperator::computeBuffer();
  _u_in_base.ownedViewOf(_u).mul_(_step_coefficient);
}

std::vector<std::vector<int>>
CoefficientTimeDerivativeOperator::stencilOffsets() const
{
  // available before the coefficient buffer is allocated
  return {std::vector<int>(_dim, 0)};
}

std::vector<FDStencilEntry>
CoefficientTimeDerivativeOperator::stencilAt(const std::vector<Real> & /*grid_spacing*/) const
{
  return {{std::vector<int>(_dim, 0),
           _w0 / _tensor_problem.subDt(),
           checkedCoefficient().contiguous()}};
}

std::vector<FDStencilEntry>
CoefficientTimeDerivativeOperator::stencilFromCoefficient(
    const torch::Tensor & coefficient, const std::vector<Real> & /*grid_spacing*/) const
{
  auto a = coefficient;
  for (const auto d : make_range(_dim))
    a = a.narrow(d, 1, a.size(d) - 2);
  return {{std::vector<int>(_dim, 0), _w0 / _tensor_problem.subDt(), a.contiguous()}};
}
