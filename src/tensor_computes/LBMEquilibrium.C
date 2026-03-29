/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMEquilibrium.h"

registerMooseObject("MarlinApp", LBMEquilibrium);

InputParameters
LBMEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute LB equilibrium distribution object.");
  params.addRequiredParam<TensorInputBufferName>(
      "bulk", "LBM bluk macroscpic parameter, e.g density or temperature");
  params.addRequiredParam<TensorInputBufferName>("velocity",
                                                 "LBM Velocty in x, y and z directions");
  return params;
}

LBMEquilibrium::LBMEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _rho(getInputBuffer("bulk", _radius)),
    _velocity(getInputBuffer("velocity", _radius))
{
}

void
LBMEquilibrium::computeBuffer()
{
  const unsigned int dim = _domain.getDim();
  const int64_t N = _u.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto vel_flat = _velocity.slice(/*dim=*/-1, /*start=*/0, /*end=*/dim).reshape({N, dim});
  auto rho_flat = _rho.reshape({N, 1});
  auto u_flat = _u.view({N, Q});
  auto w_flat = _w.view({1, Q});

  auto usqr = vel_flat.square().sum(/*dim=*/-1, /*keepdim=*/true);
  auto edotu = torch::mm(vel_flat, _e_mat.t());

  u_flat.copy_(edotu).square_().div_(2.0 * _lb_problem._cs4);
  u_flat.add_(edotu, /*alpha=*/1.0 / _lb_problem._cs2);
  u_flat.sub_(usqr, /*alpha=*/1.0 / (2.0 * _lb_problem._cs2));
  u_flat.add_(1.0);

  u_flat.mul_(w_flat);
  u_flat.mul_(rho_flat);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
