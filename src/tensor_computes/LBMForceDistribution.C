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

  torch::Tensor Fx = _forces.select(3, 0).unsqueeze(-1);
  torch::Tensor Fy = _forces.select(3, 1).unsqueeze(-1);
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
      Fz = _forces.select(3, 2).unsqueeze(-1);
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

  // Vectorized: compute all Q directions at once
  // Lattice velocities reshaped to [1,1,1,Q]
  auto ex_q = _stencil._ex.reshape({1, 1, 1, _stencil._q});
  auto ey_q = _stencil._ey.reshape({1, 1, 1, _stencil._q});
  auto ez_q = _stencil._ez.reshape({1, 1, 1, _stencil._q});

  // ci_dot_F [Nx, Ny, Nz, Q]
  auto ci_dot_F = (ex_q * Fx + ey_q * Fy + ez_q * Fz) / _lb_problem._cs2;

  // ci_dot_u and ci_dot_grad_phi [Nx, Ny, Nz, Q]
  auto ci_dot_u = ex_q * ux + ey_q * uy + ez_q * uz;
  auto ci_dot_grad_phi = ex_q * dphi_dx + ey_q * dphi_dy + ez_q * dphi_dz;

  // tensor_term = drho * (ci_dot_u * ci_dot_grad_phi) / cs2
  auto tensor_term = (drho / _lb_problem._cs2) * ci_dot_u * ci_dot_grad_phi;

  _source_term.copy_(_w * (ci_dot_F + tensor_term));
}

void
LBMForceDistribution::computeBuffer()
{
  computeSourceTerm();

  if (!_is_dynamic_relaxation)
    _u.add_(_source_term, 1.0 - 1.0 / (2.0 * _tau));
  else
  {
    if (_tau_tensor.dim() < 3)
      _tau_tensor.unsqueeze_(-1);
    _u.add_((1.0 - 1.0 / (2.0 * _tau_tensor.unsqueeze(-1))) * _source_term);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
