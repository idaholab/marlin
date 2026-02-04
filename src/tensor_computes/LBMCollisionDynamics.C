/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMCollisionDynamics.h"

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
  params.addRequiredParam<std::string>("tau0", "Relaxation parameter");
  params.addParam<std::string>("Cs", "0.1", "Relaxation parameter");
  params.addParam<bool>(
      "projection", false, "Whether or not to project non-equilibrium onto Hermite space.");

  return params;
}

template <int coll_dyn>
LBMCollisionDynamicsTempl<coll_dyn>::LBMCollisionDynamicsTempl(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f", _radius)),
    _feq(getInputBuffer("feq", _radius)),
    _tau_0(_lb_problem.getConstant<Real>(getParam<std::string>("tau0"))),
    _C_s(_lb_problem.getConstant<Real>(getParam<std::string>("Cs"))),
    _delta_x(1.0),
    _projection(getParam<bool>("projection"))
{
  //
  _shape_with_ghost = _shape;
  _shape_with_ghost[0] += 2 * _radius;
  _shape_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    _shape_with_ghost[2] += 2 * _radius;

  _fneq =
      torch::zeros({_shape_with_ghost[0], _shape_with_ghost[1], _shape_with_ghost[2], _stencil._q},
                   MooseTensor::floatTensorOptions());
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::HermiteRegularization()
{
  /**
   * Regularization procedure projects non-equilibrium (fneq) distribution
   * onto the second order Hermite space.
   */
  using torch::indexing::Slice;

  int64_t nx = _shape_with_ghost[0];
  int64_t ny = _shape_with_ghost[1];
  int64_t nz = _shape_with_ghost[2];
  auto f_flat = _f.view({nx * ny * nz, _stencil._q});
  auto feq_flat = _feq.view({nx * ny * nz, _stencil._q});
  auto f_neq_hat = _fneq.view({nx * ny * nz, _stencil._q});
  torch::Tensor fneq = f_flat - feq_flat;
  torch::Tensor fneqtimescc = torch::zeros({nx * ny * nz, 9}, MooseTensor::floatTensorOptions());
  torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic).flatten();
    fneqtimescc += (fneq.select(1, ic).view({nx * ny * nz, 1}) * ccr.view({1, 9}));
  }
  // Compute Hermite tensor
  torch::Tensor H2 = torch::zeros({1, 9}, MooseTensor::floatTensorOptions());
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic) / _lb_problem._cs2 -
                        torch::eye(3, MooseTensor::floatTensorOptions());
    H2 = ccr.flatten().unsqueeze(0).expand({nx * ny * nz, 9});

    // Compute regularized non-equilibrium distribution
    f_neq_hat.index_put_(
        {Slice(), ic},
        (_stencil._weights[ic] * (1.0 / (2.0 * _lb_problem._cs2)) * (fneqtimescc * H2).sum(1)));
  }
  _fneq = f_neq_hat.view({nx, ny, nz, _stencil._q});
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeRelaxationParameter()
{
  auto fneq_owned = ownedView(_fneq);
  auto sizes = fneq_owned.sizes();
  int64_t nx = sizes[0];
  int64_t ny = sizes[1];
  int64_t nz = sizes[2];

  auto f_neq_hat = fneq_owned.reshape({nx * ny * nz, _stencil._q, 1, 1, 1});

  torch::Tensor S;
  {
    torch::Tensor outer_products;
    {
      torch::Tensor ex_2d;
      torch::Tensor ey_2d;
      torch::Tensor ez_2d;

      {
        auto zeros = torch::zeros({_stencil._q}, MooseTensor::intTensorOptions());
        auto ones = torch::ones({_stencil._q}, MooseTensor::intTensorOptions());

        ex_2d = torch::stack({_stencil._ex, zeros, zeros});
        ey_2d = torch::stack({zeros, _stencil._ey, zeros});
        ez_2d = torch::stack({zeros, zeros, _stencil._ez});

        if (nz == 1)
          ez_2d = torch::stack({ones, zeros, _stencil._ez});
      } // zeros and ones are out of scope

      // outer product
      // expected shape: _q, 3, 3, 3
      outer_products = torch::zeros({_stencil._q, 3, 3, 3}, MooseTensor::floatTensorOptions());

      for (int i = 0; i < _stencil._q; i++)
      {
        auto ex_col = ex_2d.index({Slice(), i});
        auto ey_col = ey_2d.index({Slice(), i});
        auto ez_col = ez_2d.index({Slice(), i});
        auto outer_product = torch::einsum("i,j,k->kij", {ex_col, ey_col, ez_col});
        outer_products[i] = outer_product;
      }
      outer_products = outer_products.reshape({1, _stencil._q, 3, 3, 3});
    } // ex_2d ey_2d ez_2d are out of scope

    torch::Tensor Q_mean;
    {
      // momentum flux
      torch::Tensor Q;
      {
        // auto f_neq_outer_prod = torch::einsum("anijk,mnxyz->mijk", {outer_products, f_neq_hat});
        // until we figure out a better way to optimzie memory consumption of above commented line
        // we will do this in smaller batches
        // this will take slightly longer .... ??

        const int64_t M = nx * ny * nz;
        const int64_t batch_size = std::max<int64_t>(1, M / 20);
        torch::Tensor f_neq_outer_prod_batched =
            torch::zeros({M, 3, 3, 3}, MooseTensor::floatTensorOptions());

        for (int64_t i = 0; i < M; i += batch_size)
        {
          int64_t current_batch_size = std::min(batch_size, M - i);
          torch::Tensor f_neq_hat_batch = f_neq_hat.slice(0, i, i + current_batch_size);

          torch::Tensor batch_result =
              torch::einsum("anijk,mnxyz->mijk", {outer_products, f_neq_hat_batch});

          f_neq_outer_prod_batched.slice(0, i, i + current_batch_size).copy_(batch_result);
        }
        Q = f_neq_outer_prod_batched.reshape({nx, ny, nz, 3, 3, 3});
      }

      // mean density
      auto density = torch::sum(ownedView(_f), 3);
      auto sum_density = torch::sum(density, {0, 1, 2}).item<double>();
      auto num_points = density.numel();
      _domain.comm().sum(sum_density);
      _domain.comm().sum(num_points);
      _mean_density = sum_density / num_points;

      // Frobenius norm
      Q_mean = torch::norm(Q, 2, {3, 4, 5}) / (_mean_density * _lb_problem._cs2);
    } //  auto Q goes of scopoe

    // subgrid time scale factor
    auto t_sgs = sqrt(_C_s) * _delta_x / _lb_problem._cs;
    auto eta = _tau_0 / t_sgs;

    // mean strain rate
    auto Q_mean_sqrt = torch::sqrt(eta * eta + 4.0 * Q_mean);
    auto eta_Q_mean_sqrt = (-1.0 * eta + Q_mean_sqrt);
    S = eta_Q_mean_sqrt / (2.0 * t_sgs);
  } // outer_products is out of scope

  // relaxation parameter
  auto relaxation_owned = _tau_0 + _C_s * _delta_x * _delta_x * S / _lb_problem._cs2;
  relaxation_owned = relaxation_owned.reshape({nx, ny, nz, 1});
  _relaxation_parameter =
      torch::ones({_shape_with_ghost[0], _shape_with_ghost[1], _shape_with_ghost[2], 1},
                  MooseTensor::floatTensorOptions()) *
      _tau_0;
  auto relaxation_owned_view = ownedView(_relaxation_parameter);
  relaxation_owned_view.copy_(relaxation_owned);
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeLocalRelaxationMatrix()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    std::array<int64_t, 5> local_relaxation_mrt_shape = {
        _shape_with_ghost[0], _shape_with_ghost[1], _shape_with_ghost[2], _stencil._q, _stencil._q};

    _local_relaxation_matrix =
        torch::zeros(local_relaxation_mrt_shape, MooseTensor::floatTensorOptions());
    torch::Tensor stencil_S_expanded = _stencil._S.view({_stencil._q, _stencil._q}).clone();

    stencil_S_expanded = stencil_S_expanded.unsqueeze(0).unsqueeze(0).unsqueeze(0);
    _local_relaxation_matrix = stencil_S_expanded.expand(local_relaxation_mrt_shape).clone();
  }

  for (int64_t sh_id = 0; sh_id < _stencil._id_kinematic_visc.size(0); sh_id++)
    _local_relaxation_matrix.index_put_({Slice(),
                                         Slice(),
                                         Slice(),
                                         _stencil._id_kinematic_visc[sh_id],
                                         _stencil._id_kinematic_visc[sh_id]},
                                        1.0 / _relaxation_parameter.squeeze(-1));
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
  /* LBM BGK collision */
  _u = _feq + _fneq - 1.0 / _tau_0 * _fneq;
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<1>::MRTDynamics()
{
  computeGlobalRelaxationMatrix();

  /* LBM MRT collision */
  // f = M^{-1} x S x M x (f - feq)
  auto m_neq = torch::einsum("ab,ijkb->ijka", {_stencil._M, _fneq});
  auto m_neq_relaxed = torch::einsum("ab,ijkb->ijka", {_global_relaxation_matrix, m_neq});
  auto f = torch::einsum("ab,ijkb->ijka", {_stencil._M_inv, m_neq_relaxed});

  _u = _feq + _fneq - f;
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<2>::SmagorinskyDynamics()
{
  computeRelaxationParameter();
  // BGK collision
  _u = _feq + _fneq - 1.0 / _relaxation_parameter * _fneq;
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}

template <>
void
LBMCollisionDynamicsTempl<3>::SmagorinskyMRTDynamics()
{
  computeRelaxationParameter();
  computeLocalRelaxationMatrix();

  /* LBM MRT collision */

  auto m_neq = torch::einsum("ab,ijkb->ijka", {_stencil._M, _fneq});
  auto m_neq_relaxed = torch::einsum("ijklm,ijkm->ijkl", {_local_relaxation_matrix, m_neq});
  auto f = torch::einsum("ab,ijkb->ijka", {_stencil._M_inv, m_neq_relaxed});

  // f = M^{-1} x S x M x (f - feq)
  _u = _feq + _fneq - f;
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
    _fneq = _f - _feq;

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
