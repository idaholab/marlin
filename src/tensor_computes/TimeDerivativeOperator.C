/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "TimeDerivativeOperator.h"
#include "TensorProblem.h"

#include <limits>

registerMooseObject("MarlinApp", TimeDerivativeOperator);

InputParameters
TimeDerivativeOperator::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The buffer being time integrated");
  params.addRangeCheckedParam<unsigned int>(
      "order",
      1,
      "order = 1 | order = 2",
      "1: backward Euler. 2: BDF2 with the variable-step weights (backward Euler on the first "
      "step).");
  params.addClassDescription("Time derivative term for implicit solves, backward Euler (u/dt, "
                             "contributing u_old/dt to the right-hand side) or BDF2.");
  return params;
}

TimeDerivativeOperator::TimeDerivativeOperator(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name)),
    _u_in_base(getBufferBase(_input_name)),
    _order(getParam<unsigned int>("order")),
    _w0(1.0),
    _t_n(std::numeric_limits<Real>::quiet_NaN()),
    _last_dt(0.0),
    _dt_old(0.0)
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
  // dt / w0 is dt exactly for w0 = 1
  _u = _u_in / (_tensor_problem.subDt() / _w0);
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
  return {{std::vector<int>(_dim, 0), _w0 / _tensor_problem.subDt(), {}}};
}

torch::Tensor
TimeDerivativeOperator::rhsContribution() const
{
  const Real dt = _tensor_problem.subDt();
  if (_order == 1)
    return _u_in_base.ownedView() / dt;

  // A new start time means the last attempt was accepted: its u^n becomes u^{n-1} and its dt is
  // dt_old. A retried step starts at the same time from the same u^n and keeps the history. The
  // views alias the field's storage: the solver leaves u^n untouched (it solves in a copy and
  // restores u^n on a failure), and the next step starts from that copy.
  const Real t = _tensor_problem.subTime();
  if (t != _t_n)
  {
    if (_u_n.defined())
    {
      _u_n_minus_1 = _u_n;
      _dt_old = _last_dt;
    }
    _u_n = _u_in_base.ownedView();
    _t_n = t;
  }
  _last_dt = dt;

  // backward Euler on the first step
  if (!_u_n_minus_1.defined())
  {
    _w0 = 1.0;
    return _u_n / dt;
  }

  // MOOSE's BDF2 weights (BDF2.C); they sum to zero, so a constant is a steady state
  const Real sum = dt + _dt_old;
  _w0 = 1.0 + dt / sum;
  const Real w1 = -sum / _dt_old;
  const Real w2 = dt * dt / (_dt_old * sum);
  return -(w1 * _u_n + w2 * _u_n_minus_1) / dt;
}
