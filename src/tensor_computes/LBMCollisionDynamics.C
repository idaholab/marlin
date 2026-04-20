/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */

/**********************************************************************/
#include "LBMCollisionDynamics.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMBGKCollision);
registerMooseObject("MarlinApp", LBMMRTCollision);
registerMooseObject("MarlinApp", LBMSmagorinskyCollision);
registerMooseObject("MarlinApp", LBMSmagorinskyMRTCollision);

template <int coll_dyn>
InputParameters
LBMCollisionDynamicsTempl<coll_dyn>::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addClassDescription("Template object for LBM collision dynamics");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<TensorInputBufferName>("feq",
                                                 "Input buffer equilibrium distribution function");
  params.addParam<TensorInputBufferName>(
      "local_relaxation_matrix", "S", "Locally computed diagonal relaxation matrix");
  params.addParam<TensorInputBufferName>("tau_tensor", "tau_tensor", "Relaxation tensor");
  params.addParam<std::string>("tau0", "1.0", "Relaxation parameter");
  params.addParam<std::string>("Cs", "0.1", "Smagorinsky constant");
  params.addParam<bool>(
      "projection", false, "Whether or not to project non-equilibrium onto Hermite space.");
  params.addParam<bool>(
      "is_dynamic_relaxation", false, "Whether or not to use dynamic relaxation.");
  return params;
}

template <int coll_dyn>
LBMCollisionDynamicsTempl<coll_dyn>::LBMCollisionDynamicsTempl(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f", _radius)),
    _feq(getInputBuffer("feq", _radius)),
    _input_relaxation_matrix(getInputBuffer("local_relaxation_matrix", _radius)),
    _tau_tensor(getInputBuffer("tau_tensor", _radius)),
    _tau_0(_lb_problem.getConstant<Real>(getParam<std::string>("tau0"))),
    _C_s(_lb_problem.getConstant<Real>(getParam<std::string>("Cs"))),
    _delta_x(1.0),
    _projection(getParam<bool>("projection")),
    _is_dynamic_relaxation(getParam<bool>("is_dynamic_relaxation"))
{
  _shape_with_ghost = _shape;
  _shape_with_ghost[0] += 2 * _radius;
  _shape_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    _shape_with_ghost[2] += 2 * _radius;

  // Always pre-allocate _fneq (used in all collision operators)
  _fneq =
      torch::zeros({_shape_with_ghost[0], _shape_with_ghost[1], _shape_with_ghost[2], _stencil._q},
                   MooseTensor::floatTensorOptions());

  const int64_t N = _shape_with_ghost[0] * _shape_with_ghost[1] * _shape_with_ghost[2];

  // Hermite Pre-allocations
  if (_projection)
  {
    torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0)
                              .to(MooseTensor::floatTensorOptions());
    _C_mat = (e_xyz.t().unsqueeze(2) * e_xyz.t().unsqueeze(1)).view({_stencil._q, 9});
    auto identity_flat = torch::eye(3, MooseTensor::floatTensorOptions()).flatten().unsqueeze(0);
    torch::Tensor H2 = _C_mat / _lb_problem._cs2 - identity_flat;
    _P_mat = ((1.0 / (2.0 * _lb_problem._cs2)) * _stencil._weights.unsqueeze(0) * H2.t()).clone();

    _pi_neq_flat = torch::empty({N, 9}, MooseTensor::floatTensorOptions());
  }

  // MRT Pre-allocations
  if (coll_dyn == 1 || coll_dyn == 3)
  {
    _m_neq_flat = torch::empty({N, _stencil._q}, MooseTensor::floatTensorOptions());

    if (!_is_dynamic_relaxation && coll_dyn == 1)
    {
      computeGlobalRelaxationMatrix();
      auto MSM = torch::mm(torch::mm(_stencil._M_inv, _global_relaxation_matrix), _stencil._M);
      _MSM_t = MSM.t().clone();
    }
  }

  // Smagorinsky Pre-allocations
  if (coll_dyn == 2 || coll_dyn == 3)
  {
    int Q = _stencil._q;
    int64_t nz = _shape_with_ghost[2];
    auto zeros_q = torch::zeros({Q}, MooseTensor::intTensorOptions());
    auto ones_q = torch::ones({Q}, MooseTensor::intTensorOptions());

    auto ex_vec =
        torch::stack({_stencil._ex, zeros_q, zeros_q}, 1).to(MooseTensor::floatTensorOptions());
    auto ey_vec =
        torch::stack({zeros_q, _stencil._ey, zeros_q}, 1).to(MooseTensor::floatTensorOptions());

    torch::Tensor ez_vec;
    if (nz == 1)
      ez_vec =
          torch::stack({ones_q, zeros_q, _stencil._ez}, 1).to(MooseTensor::floatTensorOptions());
    else
      ez_vec =
          torch::stack({zeros_q, zeros_q, _stencil._ez}, 1).to(MooseTensor::floatTensorOptions());

    auto outer_products = (ex_vec.unsqueeze(2).unsqueeze(3) * ey_vec.unsqueeze(1).unsqueeze(3) *
                           ez_vec.unsqueeze(1).unsqueeze(2))
                              .permute({0, 3, 1, 2});
    _outer_flat = outer_products.reshape({Q, 27}).clone();

    _local_relaxation_parameter =
        torch::empty({_shape_with_ghost[0], _shape_with_ghost[1], _shape_with_ghost[2], 1},
                     MooseTensor::floatTensorOptions());
  }
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::HermiteRegularization()
{
  const int64_t N = _fneq.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto f_flat = _f.view({N, Q});
  auto feq_flat = _feq.view({N, Q});
  auto fneq_flat = _fneq.view({N, Q});

  torch::sub_out(fneq_flat, f_flat, feq_flat);
  torch::mm_out(_pi_neq_flat, fneq_flat, _C_mat);
  torch::mm_out(fneq_flat, _pi_neq_flat, _P_mat);
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeRelaxationParameter()
{
  const int64_t N = _fneq.numel() / _stencil._q;
  const int64_t N_owned = ownedView(_f).numel() / _stencil._q;
  const int Q = _stencil._q;

  auto fneq_flat = _fneq.view({N, Q});

  // Multiply with pre-cached outer products [N, Q] @ [Q, 27] -> [N, 27]
  auto Q_tensor_flat = torch::mm(fneq_flat, _outer_flat);

  // Mean density calculation: sum(rho) is mathematically identical to sum(f),
  // so we avoid allocating a density tensor entirely!
  auto sum_density = torch::sum(ownedView(_f)).template item<double>();
  double num_points = static_cast<double>(N_owned);

  _domain.comm().sum(sum_density);
  _domain.comm().sum(num_points);
  _mean_density = sum_density / num_points;

  // Frobenius norm of flattened tensor (Norm over the 27 components directly)
  auto Q_mean = torch::norm(Q_tensor_flat, /*p=*/2, /*dim=*/1) / (_mean_density * _lb_problem._cs2);

  auto t_sgs = sqrt(_C_s) * _delta_x / _lb_problem._cs;
  auto eta = _tau_0 / t_sgs;

  auto S = (-eta + torch::sqrt(eta * eta + 4.0 * Q_mean)) / (2.0 * t_sgs);

  auto relaxation_flat = (_tau_0 + _C_s * _delta_x * _delta_x * S / _lb_problem._cs2).unsqueeze(1);

  _local_relaxation_parameter.view({N, 1}).copy_(relaxation_flat);
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeLocalRelaxationMatrix()
{
  const int64_t N = _fneq.numel() / _stencil._q;
  const int Q = _stencil._q;

  // Initialize strictly in [N, Q, Q] shape
  if (_lb_problem.getTotalSteps() == 0)
  {
    _local_relaxation_matrix = torch::empty({N, Q, Q}, MooseTensor::floatTensorOptions());
    auto stencil_S = _stencil._S.view({Q, Q}).unsqueeze(0);
    _local_relaxation_matrix.copy_(stencil_S.expand({N, Q, Q}));
  }

  auto local_rel_flat = _local_relaxation_matrix.view({N, Q, Q});
  auto inv_tau_flat = torch::reciprocal(_local_relaxation_parameter.view({N}));

  for (int64_t sh_id = 0; sh_id < _stencil._id_kinematic_visc.size(0); sh_id++)
  {
    int64_t idx = _stencil._id_kinematic_visc[sh_id].template item<int64_t>();

    // Zero-overhead assignment replacing index_put_
    // select(2, idx) gets [N, Q], select(1, idx) gets [N]
    local_rel_flat.select(2, idx).select(1, idx).copy_(inv_tau_flat);
  }
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeGlobalRelaxationMatrix()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _global_relaxation_matrix = _stencil._S.clone();
    _global_relaxation_matrix.index_put_({_stencil._id_kinematic_visc, _stencil._id_kinematic_visc},
                                         1.0 / _tau_0);
  }
}

template <>
void
LBMCollisionDynamicsTempl<0>::BGKDynamics()
{
  _u.copy_(_feq);

  if (!_is_dynamic_relaxation)
  {
    _u.add_(_fneq, /*alpha=*/1.0 - 1.0 / _tau_0);
  }
  else
  {
    _u.add_(_fneq);
    const int64_t N = _u.numel() / _stencil._q;
    auto tau_flat = _tau_tensor.view({N, 1});
    auto u_flat = _u.view({N, _stencil._q});
    auto fneq_flat = _fneq.view({N, _stencil._q});

    u_flat.addcdiv_(fneq_flat, tau_flat, /*value=*/-1.0);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<1>::MRTDynamics()
{
  const int64_t N = _fneq.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto fneq_flat = _fneq.view({N, Q});
  auto u_flat = _u.view({N, Q});

  _u.copy_(_feq);
  _u.add_(_fneq);

  if (!_is_dynamic_relaxation)
  {
    u_flat.addmm_(fneq_flat, _MSM_t, /*beta=*/1.0, /*alpha=*/-1.0);
  }
  else
  {
    torch::mm_out(_m_neq_flat, fneq_flat, _stencil._M.t());
    _m_neq_flat.mul_(_input_relaxation_matrix.view({N, Q}));
    u_flat.addmm_(_m_neq_flat, _stencil._M_inv.t(), /*beta=*/1.0, /*alpha=*/-1.0);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<2>::SmagorinskyDynamics()
{
  computeRelaxationParameter();

  _u.copy_(_feq);
  _u.add_(_fneq);

  auto u_owned = ownedView(_u);
  auto fneq_owned = ownedView(_fneq);
  auto tau_owned = ownedView(_local_relaxation_parameter);
  u_owned.addcdiv_(fneq_owned, tau_owned, /*value=*/-1.0);

  _u_owned = u_owned;
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<3>::SmagorinskyMRTDynamics()
{
  computeRelaxationParameter();
  computeLocalRelaxationMatrix();

  const int64_t N = _fneq.numel() / _stencil._q;
  const int Q = _stencil._q;

  auto fneq_flat = _fneq.view({N, Q});

  torch::mm_out(_m_neq_flat, fneq_flat, _stencil._M.t());

  auto m_neq_expanded = _m_neq_flat.view({N, Q, 1});
  auto S_local = _local_relaxation_matrix.view({N, Q, Q});

  auto m_neq_relaxed = torch::bmm(S_local, m_neq_expanded).squeeze(-1);

  _u.copy_(_feq);
  _u.add_(_fneq);
  _u.view({N, Q}).addmm_(m_neq_relaxed, _stencil._M_inv.t(), /*beta=*/1.0, /*alpha=*/-1.0);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeBuffer()
{
  if (_projection)
    HermiteRegularization();
  else
    torch::sub_out(_fneq, _f, _feq);

  switch (coll_dyn)
  {
    case 0:
      BGKDynamics();
      break;
    case 1:
      MRTDynamics();
      break;
    case 2:
      SmagorinskyDynamics();
      break;
    case 3:
      SmagorinskyMRTDynamics();
      break;
    default:
      mooseError("Undefined template value");
  }
}

template class LBMCollisionDynamicsTempl<0>;
template class LBMCollisionDynamicsTempl<1>;
template class LBMCollisionDynamicsTempl<2>;
template class LBMCollisionDynamicsTempl<3>;
