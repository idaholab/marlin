/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMForceDistribution.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMForceDistribution);

InputParameters
LBMForceDistribution::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription(
      "Compute object for the force distribution function (source term) for phase field model..");
  params.addRequiredParam<TensorInputBufferName>("grad_phi",
                                                 "Gradient of LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("forces", "Body forces G");
  params.addParam<TensorInputBufferName>("tau_tensor", "tau_tensor", "Relaxation tensor");
  params.addRequiredParam<std::string>("tau", "Relaxation parameter tau_g");
  params.addRequiredParam<std::string>("rho_l", "Liquid density");
  params.addRequiredParam<std::string>("rho_g", "Gas density");
  params.addParam<bool>(
      "is_dynamic_relaxation", false, "Whether or not to use dynamic relaxation.");
  return params;
}

LBMForceDistribution::LBMForceDistribution(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _grad_phi(getInputBuffer("grad_phi", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _forces(getInputBuffer("forces", _radius)),
    _tau_tensor(getInputBuffer("tau_tensor", _radius)),
    _tau(_lb_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _rho_l(_lb_problem.getConstant<Real>(getParam<std::string>("rho_l"))),
    _rho_g(_lb_problem.getConstant<Real>(getParam<std::string>("rho_g"))),
    _is_dynamic_relaxation(getParam<bool>("is_dynamic_relaxation"))
{
  std::vector<int64_t> shape_q_with_ghost = _shape_q;
  shape_q_with_ghost[0] += 2 * _radius;
  shape_q_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    shape_q_with_ghost[2] += 2 * _radius;
  _source_term = torch::zeros(shape_q_with_ghost, MooseTensor::floatTensorOptions());

}

void
LBMForceDistribution::computeSourceTerm()
{
  const unsigned int & dim = _domain.getDim();

  torch::Tensor force_vec = _forces;

  torch::Tensor Fx = force_vec.select(3, 0).unsqueeze(-1);
  torch::Tensor Fy = force_vec.select(3, 1).unsqueeze(-1);
  torch::Tensor Fz;

  torch::Tensor ux = _velocity.select(3, 0).unsqueeze(-1);
  torch::Tensor uy = _velocity.select(3, 1).unsqueeze(-1);
  torch::Tensor uz;
  torch::Tensor dphi_dx = _grad_phi.select(3, 0).unsqueeze(-1);
  torch::Tensor dphi_dy = _grad_phi.select(3, 1).unsqueeze(-1);
  torch::Tensor dphi_dz;

  switch (dim)
  {
    case 3:
      Fz = force_vec.select(3, 2).unsqueeze(-1);
      uz = _velocity.select(3, 2).unsqueeze(-1);
      dphi_dz = _grad_phi.select(3, 2).unsqueeze(-1);
      break;
    case 2:
      Fz = torch::zeros_like(Fx);
      uz = torch::zeros_like(ux);
      dphi_dz = torch::zeros_like(dphi_dx);
      break;
    default:
      mooseError("Unsupported dimension for LBMForceDistribution");
  }
  const Real drho = _rho_l - _rho_g;

  for (int64_t ic = 0; ic < _stencil._q; ic++)
  {
    // c_i . (mu_phi * grad_phi + G) / cs2
    auto ci_dot_F =
        (_stencil._ex[ic] * Fx + _stencil._ey[ic] * Fy + _stencil._ez[ic] * Fz).squeeze(-1) /
        _lb_problem._cs2;

    // (rho_l - rho_g) * (u ⊗ grad_phi) : (c_i ⊗ c_i) / cs2
    // = (rho_l - rho_g) * sum_{a,b} u_a * dphi_db * c_ia * c_ib / cs2
    auto ci_dot_u = _stencil._ex[ic] * ux + _stencil._ey[ic] * uy + _stencil._ez[ic] * uz;
    auto ci_dot_grad_phi = _stencil._ex[ic] * dphi_dx + _stencil._ey[ic] * dphi_dy + _stencil._ez[ic] * dphi_dz;
    auto tensor_term = drho * (ci_dot_u * ci_dot_grad_phi).squeeze(-1) / _lb_problem._cs2;

    _source_term.index_put_({Slice(), Slice(), Slice(), ic},
                            _stencil._weights[ic] * (ci_dot_F + tensor_term));
  }
}

void
LBMForceDistribution::computeBuffer()
{
  computeSourceTerm();

  if (! _is_dynamic_relaxation)
    _u += (1.0 - 1.0 / (2.0 * _tau)) * _source_term;
  else
  {
    if (_tau_tensor.dim() < 3)
      _tau_tensor.unsqueeze_(-1);
    _u += (1.0 - 1.0 / (2.0 * _tau_tensor.unsqueeze(-1))) * _source_term;
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
