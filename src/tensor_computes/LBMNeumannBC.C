
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMNeumannBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMNeumannBC);

InputParameters
LBMNeumannBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old state distribution function");
  params.addClassDescription("LBMNeumannBC object");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution function");
  params.addRequiredParam<TensorInputBufferName>("velocity", "Fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("rho", "Fluid density");
  params.addParam<Real>("gradient",
                        "0.0"
                        "Gradient at the boundary");
  params.addParam<int>("region_id",
                       "0"
                       "Region ID for regional boundary condition");
  return params;
}

LBMNeumannBC::LBMNeumannBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _feq(getInputBuffer("feq", _radius)),
    _rho(getInputBuffer("rho", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _gradient_value(getParam<Real>("gradient"))
{
  _feq_boundary = torch::zeros_like(_feq, MooseTensor::floatTensorOptions());

  if (isParamValid("region_id") && _lb_problem.isBinaryMedia())
  {
    _region_id = getParam<int>("region_id");
    if (isBoundaryOwned(_region_id))
      _boundary_rank |= (1 << 7);
  }
  else if (!isParamValid("region_id") && _lb_problem.isBinaryMedia())
    maskBoundary();

  // Precompute specific incoming direction tensors for O(1) vectorized assignments
  auto cache_dirs =
      [&](const torch::Tensor & dirs, torch::Tensor & out_dirs, torch::Tensor & out_opps)
  {
    if (dirs.size(0) > 0)
    {
      out_dirs = dirs.to(torch::kLong);
      out_opps = _stencil._op.index_select(0, out_dirs).to(torch::kLong);
    }
  };

  cache_dirs(_stencil._left, _left_dirs, _right_dirs);
  cache_dirs(_stencil._bottom, _bottom_dirs, _top_dirs);
  cache_dirs(_stencil._front, _front_dirs, _back_dirs);
}

void
LBMNeumannBC::computeBoundaryEquilibrium()
{
  const unsigned int dim = _domain.getDim();
  auto vel_owned = ownedView(_velocity);
  auto rho_owned = ownedView(_rho);

  const int64_t N = vel_owned.numel() / vel_owned.size(-1);

  auto vel_flat = vel_owned.slice(-1, 0, dim).reshape({N, dim});
  auto rho_flat = rho_owned.reshape({N, 1});

  auto usqr = vel_flat.square().sum(-1, /*keepdim=*/true);
  auto edotu = torch::mm(vel_flat, _e_mat.t());

  auto edotu_spatial = edotu.reshape_as(_feq_boundary_owned);

  auto spatial_shape = _feq_boundary_owned.sizes().vec();
  spatial_shape.back() = 1;
  auto usqr_spatial = usqr.reshape(spatial_shape);
  auto rho_spatial = rho_flat.reshape(spatial_shape);

  _feq_boundary_owned.copy_(edotu_spatial).square_().div_(2.0 * _lb_problem._cs4);
  _feq_boundary_owned.add_(edotu_spatial, 1.0 / _lb_problem._cs2);
  _feq_boundary_owned.sub_(usqr_spatial, 1.0 / (2.0 * _lb_problem._cs2));
  _feq_boundary_owned.add_(1.0);
  _feq_boundary_owned.mul_(_w);

  // Multiply by the scalar boundary density + gradient
  _feq_boundary_owned.mul_(torch::add(rho_spatial, _gradient_value));
}

void
LBMNeumannBC::topBoundary()
{
  if (_top_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(1, _shape[1] - 1);

  auto update = _feq_boundary_owned.select(1, _shape[1] - 1).index_select(-1, _top_dirs) +
                _f_old_owned.select(1, _shape[1] - 1).index_select(-1, _top_dirs) -
                _feq_owned.select(1, _shape[1] - 1).index_select(-1, _top_dirs);

  u_face.index_copy_(-1, _top_dirs, update);
}

void
LBMNeumannBC::bottomBoundary()
{
  if (_bottom_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(1, 0);

  auto update = _feq_boundary_owned.select(1, 0).index_select(-1, _bottom_dirs) +
                _f_old_owned.select(1, 0).index_select(-1, _bottom_dirs) -
                _feq_owned.select(1, 0).index_select(-1, _bottom_dirs);

  u_face.index_copy_(-1, _bottom_dirs, update);
}

void
LBMNeumannBC::leftBoundary()
{
  if (_left_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(0, 0);

  auto update = _feq_boundary_owned.select(0, 0).index_select(-1, _left_dirs) +
                _f_old_owned.select(0, 0).index_select(-1, _left_dirs) -
                _feq_owned.select(0, 0).index_select(-1, _left_dirs);

  u_face.index_copy_(-1, _left_dirs, update);
}

void
LBMNeumannBC::rightBoundary()
{
  if (_right_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(0, _shape[0] - 1);

  auto update = _feq_boundary_owned.select(0, _shape[0] - 1).index_select(-1, _right_dirs) +
                _f_old_owned.select(0, _shape[0] - 1).index_select(-1, _right_dirs) -
                _feq_owned.select(0, _shape[0] - 1).index_select(-1, _right_dirs);

  u_face.index_copy_(-1, _right_dirs, update);
}

void
LBMNeumannBC::frontBoundary()
{
  if (_front_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(2, 0);

  auto update = _feq_boundary_owned.select(2, 0).index_select(-1, _front_dirs) +
                _f_old_owned.select(2, 0).index_select(-1, _front_dirs) -
                _feq_owned.select(2, 0).index_select(-1, _front_dirs);

  u_face.index_copy_(-1, _front_dirs, update);
}

void
LBMNeumannBC::backBoundary()
{
  if (_back_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(2, _shape[2] - 1);

  auto update = _feq_boundary_owned.select(2, _shape[2] - 1).index_select(-1, _back_dirs) +
                _f_old_owned.select(2, _shape[2] - 1).index_select(-1, _back_dirs) -
                _feq_owned.select(2, _shape[2] - 1).index_select(-1, _back_dirs);

  u_face.index_copy_(-1, _back_dirs, update);
}

void
LBMNeumannBC::wallBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == -1);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  _u_owned.index_put_({_boundary_mask},
                      _feq_boundary_owned.index({_boundary_mask}) +
                          _f_old_owned.index({_boundary_mask}) -
                          _feq_owned.index({_boundary_mask}));
}

void
LBMNeumannBC::regionalBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == _region_id);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  _u_owned.index_put_({_boundary_mask},
                      _feq_boundary_owned.index({_boundary_mask}) +
                          _f_old_owned.index({_boundary_mask}) -
                          _feq_owned.index({_boundary_mask}));
}

void
LBMNeumannBC::computeBuffer()
{
  // Prepare f_old for active domain
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  // Cache owned views for the step to avoid repeated allocations
  _feq_owned = ownedView(_feq);
  _feq_boundary_owned = ownedView(_feq_boundary);

  computeBoundaryEquilibrium();
  LBMBoundaryCondition::computeBuffer(); // Executes the boundary assignments
}
