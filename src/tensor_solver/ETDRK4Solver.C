/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ETDRK4Solver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", ETDRK4Solver);

InputParameters
ETDRK4Solver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Fourth-order exponential time differencing solver.");
  params.addParam<unsigned int>("substeps", 1, "Substeps per time step.");
  return params;
}

ETDRK4Solver::ETDRK4Solver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
  getVariables(1);
}

void
ETDRK4Solver::computeBuffer()
{
  _sub_dt = _dt / _substeps;

  auto evaluate_nonlinear = [&](const std::vector<torch::Tensor> & ubar_stage)
  {
    for (const auto i : index_range(_variables))
      _variables[i]._buffer = _domain.ifft(ubar_stage[i]);

    _compute->computeBuffer();
    forwardBuffers();

    std::vector<torch::Tensor> nonlinear(_variables.size());
    for (const auto i : index_range(_variables))
      nonlinear[i] = _variables[i]._nonlinear_reciprocal;

    return nonlinear;
  };

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    _compute->computeBuffer();
    forwardBuffers();

    std::vector<torch::Tensor> ubar_n(_variables.size());
    std::vector<torch::Tensor> linear(_variables.size());
    std::vector<torch::Tensor> nonlinear1(_variables.size());

    for (const auto i : index_range(_variables))
    {
      ubar_n[i] = _variables[i]._reciprocal_buffer;
      nonlinear1[i] = _variables[i]._nonlinear_reciprocal;

      if (_variables[i]._linear_reciprocal)
        linear[i] = *_variables[i]._linear_reciprocal;
      else
        linear[i] = torch::zeros_like(ubar_n[i]);
    }

    std::vector<torch::Tensor> ubar_b(_variables.size());
    std::vector<torch::Tensor> ubar_c(_variables.size());
    std::vector<torch::Tensor> ubar_d(_variables.size());
    std::vector<torch::Tensor> expLdt(_variables.size());
    std::vector<torch::Tensor> expHalfLdt(_variables.size());
    std::vector<torch::Tensor> phi1(_variables.size());
    std::vector<torch::Tensor> phi2(_variables.size());
    std::vector<torch::Tensor> phi3(_variables.size());

    for (const auto i : index_range(_variables))
    {
      const auto Ldt = linear[i] * _sub_dt;
      expLdt[i] = torch::exp(Ldt);
      expHalfLdt[i] = torch::exp(Ldt / 2.0);

      const auto denom = Ldt * Ldt * Ldt;
      phi1[i] = _sub_dt * (-4.0 - 3.0 * Ldt + expLdt[i] * (4.0 - Ldt)) / denom;
      phi2[i] = _sub_dt * (2.0 + Ldt + expLdt[i] * (-2.0 + Ldt)) / denom;
      phi3[i] = _sub_dt * (-4.0 - 3.0 * Ldt - Ldt * Ldt + expLdt[i] * (4.0 - Ldt)) / denom;

      const auto zero_mask = Ldt == 0.0;
      if (zero_mask.any().item<bool>())
      {
        const auto dt_tensor = torch::full_like(Ldt, _sub_dt);
        phi1[i] = torch::where(zero_mask, dt_tensor, phi1[i]);
        phi2[i] = torch::where(zero_mask, dt_tensor * dt_tensor / 2.0, phi2[i]);
        phi3[i] = torch::where(zero_mask, dt_tensor * dt_tensor / 6.0, phi3[i]);
      }

      ubar_b[i] = expHalfLdt[i] * ubar_n[i] + 0.5 * _sub_dt * nonlinear1[i];
    }

    const auto nonlinear2 = evaluate_nonlinear(ubar_b);

    for (const auto i : index_range(_variables))
      ubar_c[i] = expHalfLdt[i] * ubar_n[i] + 0.5 * _sub_dt * nonlinear2[i];

    const auto nonlinear3 = evaluate_nonlinear(ubar_c);

    for (const auto i : index_range(_variables))
      ubar_d[i] = expLdt[i] * ubar_n[i] + _sub_dt * nonlinear3[i];

    const auto nonlinear4 = evaluate_nonlinear(ubar_d);

    for (const auto i : index_range(_variables))
    {
      auto ubar = expLdt[i] * ubar_n[i] + phi1[i] * nonlinear1[i] +
                  2.0 * phi2[i] * (nonlinear2[i] + nonlinear3[i]) + phi3[i] * nonlinear4[i];

      _variables[i]._buffer = _domain.ifft(ubar);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    _sub_time += _sub_dt;
  }

  _compute->computeBuffer();
  forwardBuffers();
}
