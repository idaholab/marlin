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

  // lattice velocity components as floating point [1, 1, 1, Q]
  _e_q = {_ex.to(MooseTensor::floatTensorOptions()),
          _ey.to(MooseTensor::floatTensorOptions()),
          _ez.to(MooseTensor::floatTensorOptions())};
}

void
LBMForceDistribution::computeSourceTerm()
{
  const unsigned int dim = _domain.getDim();
  if (dim != 2 && dim != 3)
    mooseError("Unsupported dimension for LBMForceDistribution");

  // c_i . v for all Q directions, accumulated in place in the original order
  // (e_x v_x + e_y v_y) + e_z v_z. The lattice components are -1, 0 or 1, so every product is
  // exact and addcmul_ rounds exactly like the separate multiply + add.
  const auto ci_dot = [&](const torch::Tensor & v)
  {
    auto r = _e_q[0] * v.select(3, 0).unsqueeze(-1);
    r.addcmul_(_e_q[1], v.select(3, 1).unsqueeze(-1));
    if (dim == 3)
      r.addcmul_(_e_q[2], v.select(3, 2).unsqueeze(-1));
    return r;
  };

  const Real drho = _rho_l - _rho_g;

  // ci_dot_F / cs2  [Nx, Ny, Nz, Q]
  auto source = ci_dot(_forces).div_(_lb_problem._cs2);

  // tensor_term = drho / cs2 * (ci_dot_u * ci_dot_grad_phi)
  source.add_(ci_dot(_velocity).mul_(drho / _lb_problem._cs2).mul_(ci_dot(_grad_phi)));

  torch::mul_out(_source_term, _w, source);
}

void
LBMForceDistribution::computeBuffer()
{
  computeSourceTerm();

  if (!_is_dynamic_relaxation)
    _u.add_(_source_term, 1.0 - 1.0 / (2.0 * _tau));
  else
  {
    // local view: do not change the shape of the shared tau buffer
    const auto tau = _tau_tensor.dim() < 3 ? _tau_tensor.unsqueeze(-1) : _tau_tensor;
    _u.add_((1.0 - 1.0 / (2.0 * tau.unsqueeze(-1))) * _source_term);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, _solid_values);
}
