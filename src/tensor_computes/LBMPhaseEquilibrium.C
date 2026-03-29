/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMPhaseEquilibrium.h"

registerMooseObject("MarlinApp", LBMPhaseEquilibrium);

InputParameters
LBMPhaseEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription(
      "Compute LB equilibrium distribution object for phase field parameter.");
  params.addRequiredParam<TensorInputBufferName>("phi", "LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");

  return params;
}

LBMPhaseEquilibrium::LBMPhaseEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi", _radius)),
    _velocity(getInputBuffer("velocity", _radius))
{
}

void
LBMPhaseEquilibrium::computeBuffer()
{
  const unsigned int dim = _domain.getDim();
  const int64_t N = _u.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto vel_flat = _velocity.slice(-1, 0, dim).reshape({N, dim});
  auto phi_flat = _phi.reshape({N, 1});
  auto u_flat = _u.view({N, Q});

  // edotu = vel_flat @ _e_mat.t()
  torch::mm_out(u_flat, vel_flat, _e_mat.t());

  u_flat.div_(_lb_problem._cs2);
  u_flat.add_(1.0);
  u_flat.mul_(_w.view({1, Q}));
  u_flat.mul_(phi_flat);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
