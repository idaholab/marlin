/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AdamsBashforthMoultonCoupled.h"
#include "TensorProblem.h"
#include "DomainAction.h"

#include <array>

registerMooseObject("MarlinApp", AdamsBashforthMoultonCoupled);

InputParameters
AdamsBashforthMoultonCoupled::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Coupled Adams-Bashforth-Moulton solver with dense linear operator "
                             "and batched torch solve in reciprocal space.");

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
      "corrector_steps", 0, "Number of Adams-Moulton corrector steps (0 disables the corrector).");

  // Off-diagonal linear operator specification
  params.addParam<std::vector<unsigned int>>("linear_offdiag_rows", {}, "Row indices for L_ij.");
  params.addParam<std::vector<unsigned int>>("linear_offdiag_cols", {}, "Column indices for L_ij.");
  params.addParam<std::vector<TensorInputBufferName>>(
      "linear_offdiag", {}, "Off-diagonal linear operator buffers.");
  params.addParam<bool>("assume_symmetric",
                        false,
                        "Mirror off-diagonal entries (i,j) into (j,i) if not explicitly provided.");

  return params;
}

AdamsBashforthMoultonCoupled::AdamsBashforthMoultonCoupled(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _predictor_order(getParam<std::size_t>("predictor_order") - 1),
    _corrector_order(getParam<std::size_t>("corrector_order") - 1),
    _corrector_steps(getParam<std::size_t>("corrector_steps")),
    _assume_symmetric(getParam<bool>("assume_symmetric")),
    _L_offdiag_indices(
        getParam<unsigned int, unsigned int>("linear_offdiag_rows", "linear_offdiag_cols")),
    _L_offdiag_names(getParam<std::vector<TensorInputBufferName>>("linear_offdiag")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
  // request history consistent with chosen orders
  const auto history = std::max(_predictor_order, _corrector_order);
  getVariables(history);

  if (_L_offdiag_indices.size() != _L_offdiag_names.size())
    paramError("linear_offdiag",
               "'linear_offdiag_rows', 'linear_offdiag_cols', and 'linear_offdiag' must all have "
               "the same length.");

  const auto N = _variables.size();
  for (const auto & [i, j] : _L_offdiag_indices)
  {
    if (i >= N)
      paramError("linear_offdiag_rows", "Off-diagonal indices out of range.");
    if (i >= N)
      paramError("linear_offdiag_cols", "Off-diagonal indices out of range.");
  }

  for (const auto & name : _L_offdiag_names)
    _L_offdiag_buffer.push_back(&getInputBufferByName(name));
}

void
AdamsBashforthMoultonCoupled::computeBuffer()
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
  _sub_dt = _dt / _substeps;

  const auto N = _variables.size();
  if (N == 0)
    return;

  // basic shape/dtype/device helpers from the first variable
  const auto & base_ubar = *_variables[0]._linear_reciprocal;
  const auto base_opts = base_ubar.options();
  const auto base_dtype = base_ubar.scalar_type();

  // Pre-construct a zeros tensor for missing L entries
  const auto zeros_like_grid = torch::zeros_like(base_ubar);

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // re-evaluate the solve compute
    _compute->computeBuffer();
    forwardBuffers();

    // Predictor: build rhs per variable (ABm)
    std::vector<torch::Tensor> rhs_list(N);
    for (const auto i : make_range(N))
    {
      const auto & reciprocal_buffer = _variables[i]._reciprocal_buffer;
      const auto & nonlinear_reciprocal = _variables[i]._nonlinear_reciprocal;
      const auto & old_nonlinear_reciprocal = _variables[i]._old_nonlinear_reciprocal;

      const auto n_old = old_nonlinear_reciprocal.size();
      const auto order =
          std::min(substep < _predictor_order && dt_changed ? 0 : n_old, _predictor_order);

      auto rhs = reciprocal_buffer + (_sub_dt * beta[order][0]) * nonlinear_reciprocal;
      for (const auto j : make_range(order))
        rhs += (_sub_dt * beta[order][j + 1]) * old_nonlinear_reciprocal[j];

      rhs_list[i] = rhs;
    }

    // Assemble L as dense [grid..., N, N] by stacking rows
    // map off-diagonal entries into a table of pointers for quick access
    std::vector<const torch::Tensor *> Lptr(N * N, &zeros_like_grid);

    // diagonal from linear_reciprocal (may be null meaning zero)
    for (const auto i : make_range(N))
      Lptr[i * N + i] = _variables[i]._linear_reciprocal; // may remain null

    // off-diagonals
    for (const auto k : index_range(_L_offdiag_buffer))
    {
      const auto & [i, j] = _L_offdiag_indices[k];
      Lptr[i * N + j] = _L_offdiag_buffer[k];
      if (_assume_symmetric && i != j && Lptr[j * N + i] == &zeros_like_grid)
        Lptr[j * N + i] = _L_offdiag_buffer[k];
    }

    using torch::stack;
    std::vector<torch::Tensor> rows;
    rows.reserve(N);
    for (const auto i : make_range(N))
    {
      std::vector<torch::Tensor> cols;
      cols.reserve(N);
      for (const auto j : make_range(N))
        cols.push_back(*Lptr[i * N + j]);
      // [grid..., N]
      rows.push_back(stack(cols, -1));
    }

    // L [grid..., N, N]
    auto L = stack(rows, -1);
    // A = I - dt * L (cast to match rhs dtype)
    auto I = torch::eye(N, base_opts);
    auto A = I - _sub_dt * L.to(base_dtype);
    // rhs [grid..., N]
    auto b = stack(rhs_list, -1).to(base_dtype);

    // Solve A * ubar = b (batched over grid points)
    // Broadcast I to grid dims is automatic in linalg_solve since A has those dims
    const auto ubar_all = at::linalg_solve(A, b, true);

    // Update physical-space variables via inverse FFT
    auto ubar_solutions = torch::unbind(ubar_all, -1);
    for (const auto i : make_range(N))
      _variables[i]._buffer = _domain.ifft(ubar_solutions[i]);

    // advance time
    _sub_time += _sub_dt;

    // Corrector (optional)
    if (_corrector_steps)
    {
      // snapshot ubar at t_n
      std::vector<torch::Tensor> ubar_n(N);
      for (const auto i : make_range(N))
        ubar_n[i] = _variables[i]._reciprocal_buffer;

      // N at time n (from AB predictor call above)
      std::vector<torch::Tensor> N_n(N);
      if (_corrector_order > 0)
        for (const auto i : make_range(N))
          N_n[i] = _variables[i]._nonlinear_reciprocal;

      for (std::size_t j_corr = 0; j_corr < _corrector_steps; ++j_corr)
      {
        // recompute nonlinearity with current predicted values
        _compute->computeBuffer();
        forwardBuffers();

        // Build corrector RHS
        std::vector<torch::Tensor> rhs_corr(N);
        for (const auto i : make_range(N))
        {
          const auto & nonlinear_pred = _variables[i]._nonlinear_reciprocal;
          const auto & old_nonlinear = _variables[i]._old_nonlinear_reciprocal;
          const auto n_old = old_nonlinear.size();

          const auto order =
              std::min(substep < _corrector_order && dt_changed ? 1 : n_old + 1, _corrector_order);
          if (order == 0)
          {
            rhs_corr[i] = ubar_n[i];
            continue;
          }

          auto rhs = ubar_n[i] + (_sub_dt * alpha[order][0]) * nonlinear_pred;
          if (order > 0)
          {
            rhs += (_sub_dt * alpha[order][1]) * N_n[i];
            for (const auto jj : make_range(order - 1))
              rhs += (_sub_dt * alpha[order][jj + 2]) * old_nonlinear[jj];
          }
          rhs_corr[i] = rhs;
        }

        // Re-assemble L (allowing time dependence)
        std::vector<const torch::Tensor *> Lptr_c(N * N, &zeros_like_grid);
        for (const auto i : make_range(N))
          Lptr_c[i * N + i] = _variables[i]._linear_reciprocal;
        for (const auto k : index_range(_L_offdiag_names))
        {
          const auto & [i, j] = _L_offdiag_indices[k];
          Lptr_c[i * N + j] = _L_offdiag_buffer[k];
          if (_assume_symmetric && i != j && Lptr_c[j * N + i] == &zeros_like_grid)
            Lptr_c[j * N + i] = _L_offdiag_buffer[k];
        }

        std::vector<torch::Tensor> rows_c;
        rows_c.reserve(N);
        for (const auto i : make_range(N))
        {
          std::vector<torch::Tensor> cols;
          cols.reserve(N);
          for (const auto j : make_range(N))
            cols.push_back(*Lptr_c[i * N + j]);
          rows_c.push_back(stack(cols, -1));
        }

        auto Lc = stack(rows_c, -1);
        auto Ic = torch::eye(N, base_opts);
        auto Ac = Ic - _sub_dt * Lc.to(base_dtype);
        auto bc = stack(rhs_corr, -1).to(base_dtype);

        const auto ubar_all_corr = at::linalg_solve(Ac, bc, true);
        auto ubar_corr_list = torch::unbind(ubar_all_corr, -1);
        for (const auto i : make_range(N))
          _variables[i]._buffer = _domain.ifft(ubar_corr_list[i]);
      }
    }

    // skip final advance (MOOSE will do it)
    if (substep < _substeps - 1)
      _tensor_problem.advanceState();
  }
}
