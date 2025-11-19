/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AdamsBashforthMoulton.h"
#include "TensorProblem.h"
#include "Conversion.h"
#include "DomainAction.h"
#include <array>

registerMooseObject("MarlinApp", AdamsBashforthMoulton);
registerMooseObjectRenamed("MarlinApp",
                           SemiImplicitSolver,
                           "10/01/2025 00:01",
                           AdamsBashforthMoulton);

InputParameters
AdamsBashforthMoulton::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Adams-Bashforth-Moulton semi-implicit/explicit time integration "
                             "solver with optional implicit corrector.");
  params.addParam<unsigned int>("substeps", 1, "semi-implicit substeps per time step.");
  params.addRangeCheckedParam<std::size_t>("predictor_order",
                                           2,
                                           "predictor_order > 0 & predictor_order <= " +
                                               Moose::stringify(max_order),
                                           "Order of the Adams-Bashforth predictor.");
  params.addRangeCheckedParam<std::size_t>("corrector_order",
                                           2,
                                           "corrector_order > 0 & corrector_order <= " +
                                               Moose::stringify(max_order),
                                           "Order of the Adams-Moulton corrector.");
  params.addParam<std::size_t>(
      "corrector_steps",
      0,
      "Number the Adams-Moulton corrector steps to take (one is usually sufficient).");
  return params;
}

AdamsBashforthMoulton::AdamsBashforthMoulton(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _predictor_order(getParam<std::size_t>("predictor_order") - 1),
    _corrector_order(getParam<std::size_t>("corrector_order") - 1),
    _corrector_steps(getParam<std::size_t>("corrector_steps")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
  // request history consistent with chosen orders
  const auto history = std::max(_predictor_order, _corrector_order);
  getVariables(history);
}

void
AdamsBashforthMoulton::computeBuffer()
{
  // Adams–Bashforth coefficients (zero-padded)
  constexpr std::array<std::array<double, max_order>, max_order> beta = {{
      {1.0, 0.0, 0.0, 0.0, 0.0},                                                        // AB1
      {3.0 / 2.0, -1.0 / 2.0, 0.0, 0.0, 0.0},                                           // AB2
      {23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0, 0.0, 0.0},                                // AB3
      {55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0, 0.0},                       // AB4
      {190.0 / 720.0, -2774.0 / 720.0, 2616.0 / 720.0, -1274.0 / 720.0, 251.0 / 720.0}, // AB5
  }};

  // Adams–Moulton coefficients (zero-padded)
  constexpr std::array<std::array<double, max_order>, max_order> alpha = {{
      {1.0, 0.0, 0.0, 0.0, 0.0},                                                    // AM1
      {0.5, 0.5, 0.0, 0.0, 0.0},                                                    // AM2
      {5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0, 0.0, 0.0},                              // AM3
      {9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0, 0.0},                      // AM4
      {251.0 / 720.0, 646.0 / 720.0, -264.0 / 720.0, 106.0 / 720.0, -19.0 / 720.0}, // AM5
  }};

  const bool dt_changed = (_dt != _dt_old);

  torch::Tensor ubar;
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // re-evaluate the solve compute
    _compute->computeBuffer();
    forwardBuffers();

    // Adams-Bashforth predictor on all variables
    for (auto & [u,
                 reciprocal_buffer,
                 linear_reciprocal,
                 nonlinear_reciprocal,
                 old_nonlinear_reciprocal] : _variables)
    {
      const auto n_old = old_nonlinear_reciprocal.size();

      // Order is what the user requested, or what the available history allows for.
      // If dt changes between steps, we start at first order again
      const auto order =
          std::min(substep < _predictor_order && dt_changed ? 0 : n_old, _predictor_order);

      // Adams-Bashforth
      ubar = reciprocal_buffer + (_sub_dt * beta[order][0]) * nonlinear_reciprocal;
      for (const auto i : make_range(order))
        ubar += (_sub_dt * beta[order][i + 1]) * old_nonlinear_reciprocal[i];

      if (linear_reciprocal)
        ubar /= (1.0 - _sub_dt * *linear_reciprocal);

      u = _domain.ifft(ubar);
    }

    // AB: y[n+1] = y[n] + dt * f(y[n])
    // AM: y[n+1] = y[n] + dt * f(y[n+1])

    // increment substep time
    _sub_time += _sub_dt;

    // Adams-Moulton corrector
    if (_corrector_steps)
    {
      // we need to keep the previous time step reciprocal_buffer if we run the AM corrector
      std::vector<torch::Tensor> ubar_n(_variables.size());
      for (const auto k : index_range(_variables))
        ubar_n[k] = _variables[k]._reciprocal_buffer;

      // if the corrector order is AM2 or higher we also need the f calculated by AB prior to the
      // update to the current step values
      std::vector<torch::Tensor> N_n;
      if (_corrector_order > 0)
      {
        N_n.resize(_variables.size());
        for (const auto k : index_range(_variables))
          N_n[k] = _variables[k]._nonlinear_reciprocal;
      }

      // apply multiple corrector steps, going forward we probably want to allow users to fixpoint
      // iterate until a given convergence criterion is fulfilled.
      for (std::size_t j = 0; j < _corrector_steps; ++j)
      {
        // re-evaluate the solve compute with the predicted variable values
        _compute->computeBuffer();
        forwardBuffers();

        for (const auto k : index_range(_variables))
        {
          auto & u = _variables[k]._buffer;
          const auto * linear_reciprocal = _variables[k]._linear_reciprocal;
          const auto & nonlinear_reciprocal_pred = _variables[k]._nonlinear_reciprocal;
          const auto & old_nonlinear_reciprocal = _variables[k]._old_nonlinear_reciprocal;

          const auto n_old = old_nonlinear_reciprocal.size();
          const auto order =
              std::min(substep < _corrector_order && dt_changed ? 1 : n_old + 1, _corrector_order);
          if (order == 0)
            continue;

          ubar = ubar_n[k] + (_sub_dt * alpha[order][0]) * nonlinear_reciprocal_pred;

          if (order > 0)
          {
            ubar += (_sub_dt * alpha[order][1]) * N_n[k];
            for (const auto i : make_range(order - 1))
              ubar += (_sub_dt * alpha[order][i + 2]) * old_nonlinear_reciprocal[i];
          }

          if (linear_reciprocal)
            ubar /= (1.0 - _sub_dt * *linear_reciprocal);

          u = _domain.ifft(ubar);
        }
      }
    }

    // we skip the advanceState on the last substep because MOOSE will call that automatically
    if (substep < _substeps - 1)
      _tensor_problem.advanceState();
  }
}
