/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeEffectiveRelaxation.h"

registerMooseObject("MarlinApp", LBMComputeEffectiveRelaxation);

InputParameters
LBMComputeEffectiveRelaxation::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addRequiredParam<TensorInputBufferName>("local_pore_size", "Local pore size");
  params.addRequiredParam<TensorInputBufferName>("local_Knudsen_number", "Local Knudsen number");
  params.addParam<std::string>("mfp", "1.0e-9", "Mean free path of the system, (meters)");
  params.addParam<std::string>("dx", "1.0e-9", "Domain resolution, (meters)");
  params.addParam<std::string>("A2", "0.8", "Second order slip boundary constant");

  params.addClassDescription("Compute local effective relaxation time matrix based on local pore "
                             "size and Knudsen number.");

  return params;
}

LBMComputeEffectiveRelaxation::LBMComputeEffectiveRelaxation(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _local_pore_size(getInputBuffer("local_pore_size", _radius)),
    _local_Knudsen_number(getInputBuffer("local_Knudsen_number", _radius)),
    _mfp(_lb_problem.getConstant<Real>(getParam<std::string>("mfp"))),
    _dx(_lb_problem.getConstant<Real>(getParam<std::string>("dx"))),
    _A2(_lb_problem.getConstant<Real>(getParam<std::string>("A2")))
{
  if (_stencil._q != 9)
    mooseError("LBMComputeEffectiveRelaxation currently only supports D2Q9 stencils.");

  // static scalar coefficients
  _C1 = std::sqrt(6.0 / libMesh::pi);
  _C2 = 3.0 * std::sqrt(3.0) / 8.0 * (_mfp / _dx);
}

void
LBMComputeEffectiveRelaxation::computeBuffer()
{
  const int64_t N = _u.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto u_flat = _u.view({N, Q});
  auto pore_flat = _local_pore_size.view({N, 1});
  auto kn_flat = _local_Knudsen_number.view({N, 1});

  auto denom = kn_flat * 2.0 + 1.0;
  auto tau_s = (pore_flat * kn_flat).div_(denom).mul_(_C1).add_(0.5);

  auto ts_diff = tau_s * 2.0 - 1.0;
  auto tau_q = ts_diff.square().mul_(libMesh::pi * _A2).add_(3.0).div_(ts_diff * 8.0).add_(0.5);
  auto tau_d = torch::reciprocal(denom).mul_(_C2).add_(0.5);

  auto inv_tau_s = torch::reciprocal(tau_s);
  auto inv_tau_q = torch::reciprocal(tau_q);
  auto inv_tau_d = torch::reciprocal(tau_d);

  // Safely copy [N] to [N]
  u_flat.select(1, 0).fill_(1.0);
  u_flat.select(1, 1).fill_(1.0 / 1.1);
  u_flat.select(1, 2).fill_(1.0 / 1.2);
  u_flat.select(1, 3).copy_(inv_tau_d.squeeze(-1));
  u_flat.select(1, 4).copy_(inv_tau_q.squeeze(-1));
  u_flat.select(1, 5).copy_(inv_tau_d.squeeze(-1));
  u_flat.select(1, 6).copy_(inv_tau_q.squeeze(-1));
  u_flat.select(1, 7).copy_(inv_tau_s.squeeze(-1));
  u_flat.select(1, 8).copy_(inv_tau_s.squeeze(-1));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
