/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "BroydenSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", BroydenSolver);

InputParameters
BroydenSolver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Implicit secant solver time integration.");
  params.addParam<unsigned int>("substeps", 1, "secant solver substeps per time step.");
  params.addParam<unsigned int>("max_iterations", 5, "Maximum number of secant solver iteration.");
  params.addParam<Real>("relative_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("absolute_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("damping", 1.0, "Damping factor for the update step.");
  params.addParam<Real>(
      "initial_jacobian_guess", 1.0, "Factor for the initial inverse jacobian guess.");
  params.addParam<Real>(
      "dt_epsilon", 1e-4, "Semi-implicit stable timestep to bootstrap secant solve.");
  params.set<unsigned int>("substeps") = 0;
  params.addParam<bool>("verbose", false, "Show convergence history.");
  return params;
}

BroydenSolver::BroydenSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    IterativeTensorSolverInterface(),
    _substeps(getParam<unsigned int>("substeps")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _relative_tolerance(getParam<Real>("relative_tolerance")),
    _absolute_tolerance(getParam<Real>("absolute_tolerance")),
    _verbose(getParam<bool>("verbose")),
    _damping(getParam<Real>("damping")),
    _eye_factor(getParam<Real>("initial_jacobian_guess")),
    _dim(_domain.getDim()),
    _options(MooseTensor::complexFloatTensorOptions())
{
  // no history required
  getVariables(0);

  // Jacobian dimensions
  const auto n = _variables.size();
  const auto & s = _domain.getReciprocalShape();
  std::vector<int64_t> v(s.begin(), s.end());
  v.push_back(n);
  v.push_back(n);

  // Create a 3x3 identity matrix and expand to all grid points
  _M = torch::eye(n, _options) * _eye_factor;
  for (const auto i : make_range(_dim))
  {
    libmesh_ignore(i);
    _M.unsqueeze_(0);
  }
  _M = _M.expand(v);
}

void
BroydenSolver::substep()
{
  // initial residual
  _compute->computeBuffer();
  forwardBuffers();

  const auto n = _variables.size();

  // stack u_old
  std::vector<torch::Tensor> u_old_v(n);
  for (const auto i : make_range(n))
    if (_variables[i]._reciprocal_buffer.defined())
      u_old_v[i] = _variables[i]._reciprocal_buffer;
    else
      u_old_v[i] = _domain.fft(_variables[i]._buffer);
  const auto u_old = torch::stack(u_old_v, -1);

  auto stackVariables = [&]()
  {
    std::vector<torch::Tensor> u(n);
    std::vector<torch::Tensor> N(n);
    std::vector<torch::Tensor> L(n);
    for (const auto i : make_range(n))
    {
      u[i] = _variables[i]._reciprocal_buffer;
      N[i] = _variables[i]._nonlinear_reciprocal;
      L[i] =
          _variables[i]._linear_reciprocal ? *(_variables[i]._linear_reciprocal) : torch::tensor(0);
    }
    return std::make_tuple(torch::stack(u, -1), torch::stack(N, -1), torch::stack(L, -1));
  };

  const auto [u0, N, L] = stackVariables();
  torch::Tensor u = u0;
  torch::Tensor R = (N + L * u) * _sub_dt;

  // initial residual norm
  const auto R0norm = torch::norm(R).item<double>();

  // secant iterations
  for (_iterations = 0; _iterations < _max_iterations; ++_iterations)
  {
    // check for convergence
    const auto Rnorm = torch::norm(R).item<double>();

    // NaN check
    if (std::isnan(Rnorm))
      mooseError("NAN!");

    // residual divergence check
    if (_iterations > 4 && Rnorm * 10.0 / _iterations > R0norm)
      mooseWarning("Diverging residual ", Rnorm, " ", Rnorm * 10.0 / _iterations, ' ', R0norm);

    if (Rnorm < _absolute_tolerance || Rnorm / R0norm < _relative_tolerance)
    {
      std::cout << "Broyden solve converged after " << _iterations << " iterations. |R|=" << Rnorm
                << " |R|/|R0|=" << Rnorm / R0norm << '\n';
      _is_converged = true;
      return;
    }
    else if (_verbose)
      std::cout << _iterations << " |R|=" << Rnorm << std::endl;

    // update step dx
    const auto sk = -torch::matmul(_M, R.unsqueeze(-1)); // column vector
    const auto skT = sk.squeeze(-1).unsqueeze(-2);      // row vector

    // update u
    const auto u_out_v = torch::unbind(u + sk.squeeze(-1) * 0.5, -1);
    for (const auto i : make_range(n))
    {
      // look at min max here and maybe apply bounds?
      _variables[i]._buffer = _domain.ifft(u_out_v[i]);
    }

    // update residual
    _compute->computeBuffer();
    forwardBuffers();

    const auto [u0, N, L] = stackVariables();
    u = u0;
    const auto Rnew = (N + L * u) * _sub_dt + u_old - u;

    // residual change
    const auto yk = (Rnew - R).unsqueeze(-1);
    const auto ykT = yk.squeeze(-1).unsqueeze(-2);

    const auto denom = torch::matmul(skT, yk);
    if (_verbose)
    {
      const auto denom_zero = torch::sum(torch::where(torch::abs(denom) == 0, 1, 0)).item<double>();
      if (denom_zero)
        std::cout << "Matrix update denominator is zero in " << denom_zero << " entries.\n";
    }

    _M = _M + torch::where(torch::abs(denom) > 1e-12,
                           torch::matmul((sk - torch::matmul(_M, yk)), skT) / denom,
                           0.0);
    // _M = _M + torch::matmul((sk - torch::matmul(_M, yk)), skT) /
    //             torch::where(torch::abs(denom) > 1e-12, denom, 1.0);

    // _M = _M + torch::matmul((sk - torch::matmul(_M, yk)), skT) / torch::matmul(skT, yk);

    R = Rnew;
  }

  std::cerr << "Broyden solve did not converge within the maximum number of iterations.\n";
  _is_converged = false;
}
