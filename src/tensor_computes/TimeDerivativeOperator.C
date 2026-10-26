/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "TimeDerivativeOperator.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", TimeDerivativeOperator);

InputParameters
TimeDerivativeOperator::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The buffer being time integrated");
  params.addClassDescription("Backward Euler time derivative term u/dt for implicit solves, "
                             "contributing u_old/dt to the right-hand side.");
  return params;
}

TimeDerivativeOperator::TimeDerivativeOperator(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name)),
    _u_in_base(getBufferBase(_input_name))
{
}

void
TimeDerivativeOperator::realSpaceComputeBuffer()
{
  computeBuffer();
}

void
TimeDerivativeOperator::computeBuffer()
{
  _u = _u_in / _tensor_problem.subDt();
}

std::vector<FDStencilEntry>
TimeDerivativeOperator::stencilAt() const
{
  return stencilAt({});
}

std::vector<FDStencilEntry>
TimeDerivativeOperator::stencilAt(const std::vector<Real> & /*grid_spacing*/) const
{
  // a single centre entry, independent of the grid spacing
  return {{std::vector<int>(_dim, 0), 1.0 / _tensor_problem.subDt(), {}}};
}

torch::Tensor
TimeDerivativeOperator::rhsContribution() const
{
  return _u_in_base.ownedView() / _tensor_problem.subDt();
}
