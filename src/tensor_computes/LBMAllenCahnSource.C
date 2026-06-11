/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMAllenCahnSource.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMAllenCahnSource);

InputParameters
LBMAllenCahnSource::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute Allen-Cahn source term for phase field model.");
  params.addRequiredParam<TensorInputBufferName>("phi", "LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("grad_phi",
                                                 "Gradient of LBM phase field parameter");
  params.addRequiredParam<std::string>("tau", "Relaxation parameter for LBM phase field");
  params.addRequiredParam<std::string>("thickness", "Interface thickness");

  return params;
}

LBMAllenCahnSource::LBMAllenCahnSource(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _grad_phi(getInputBuffer("grad_phi", _radius)),
    _tau(_lb_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _D(_lb_problem.getConstant<Real>(getParam<std::string>("thickness")))
{
  std::vector<int64_t> shape_q_with_ghost = _shape_q;
  shape_q_with_ghost[0] += 2 * _radius;
  shape_q_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    shape_q_with_ghost[2] += 2 * _radius;

  _source_term = torch::zeros(shape_q_with_ghost, MooseTensor::floatTensorOptions());

  // Precompute the projection matrix: _P_mat = (weights / cs2) * E
  std::vector<torch::Tensor> e_vec = {_stencil._ex, _stencil._ey};
  if (_dim == 3)
    e_vec.push_back(_stencil._ez);

  torch::Tensor E_mat =
      torch::stack(e_vec, /*dim=*/0).to(MooseTensor::floatTensorOptions()); // [dim, Q]
  auto w_flat = _stencil._weights.unsqueeze(0);                             // [1, Q]

  _P_mat = (E_mat * (w_flat / _lb_problem._cs2)).clone();
}

void
LBMAllenCahnSource::computeSourceTerm()
{
  const unsigned int dim = _domain.getDim();
  const int64_t N = _velocity.numel() / _velocity.size(-1);
  const int Q = _stencil._q;

  auto phi_flat = _phi.view({N, 1});
  auto vel_flat = _velocity.slice(-1, 0, dim).reshape({N, dim});
  auto grad_flat = _grad_phi.slice(-1, 0, dim).reshape({N, dim});
  auto source_flat = _source_term.view({N, Q});

  // Initialize flat
  if (_phi_u_old.numel() == 0)
    _phi_u_old = torch::zeros({N, dim}, MooseTensor::floatTensorOptions());
  auto phi_u_old_flat = _phi_u_old.view({N, dim});

  auto phi_u = phi_flat * vel_flat;
  torch::Tensor A = torch::sub(phi_u, phi_u_old_flat);

  auto mag = torch::norm(grad_flat, 2, -1, /*keepdim=*/true); // [N, 1]

  auto lambda_factor = 1.0 - phi_flat;
  lambda_factor.mul_(phi_flat);
  lambda_factor.mul_(_lb_problem._cs2 * 4.0 / _D);
  lambda_factor.div_(mag + 1.0e-16);

  A.addcmul_(lambda_factor, grad_flat);
  torch::mm_out(source_flat, A, _P_mat);

  phi_u_old_flat.copy_(phi_u);
}

void
LBMAllenCahnSource::computeBuffer()
{
  computeSourceTerm();

  const int64_t N = _u.numel() / _stencil._q;
  auto u_flat = _u.view({N, _stencil._q});
  auto source_flat = _source_term.view({N, _stencil._q});

  u_flat.add_(source_flat, /*alpha=*/1.0 - 1.0 / (2.0 * _tau));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
