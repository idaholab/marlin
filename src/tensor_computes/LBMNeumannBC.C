
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

torch::Tensor
LBMNeumannBC::boundaryEquilibrium(const torch::Tensor & velocity, const torch::Tensor & rho)
{
  // equilibrium at the given nodes only (leading dims of velocity), shape [..., Q]
  const unsigned int dim = _domain.getDim();
  const int64_t N = velocity.numel() / velocity.size(-1);
  const int64_t Q = _stencil._q;
  auto vel_flat = velocity.slice(-1, 0, dim).reshape({N, dim});

  // macroscopic variables
  auto usqr = vel_flat.square().sum(-1, /*keepdim=*/true);
  auto edotu = torch::mm(vel_flat, _e_mat.t());

  // feq_b = w * rho_bc * (1 + edotu/cs2 + edotu^2/2cs4 - usqr/2cs2)
  auto feq_b = torch::square(edotu).div_(2.0 * _lb_problem._cs4);
  feq_b.add_(edotu, 1.0 / _lb_problem._cs2);
  feq_b.sub_(usqr, 1.0 / (2.0 * _lb_problem._cs2));
  feq_b.add_(1.0);
  feq_b.mul_(_w.view({1, Q}));

  // Multiply by the scalar boundary density + gradient
  feq_b.mul_(torch::add(rho.reshape({N, 1}), _gradient_value));

  auto shape = velocity.sizes().vec();
  shape.back() = Q;
  return feq_b.view(shape);
}

torch::Tensor
LBMNeumannBC::faceEquilibrium(int64_t dim, int64_t index)
{
  return boundaryEquilibrium(ownedView(_velocity).select(dim, index), ownedView(_rho).select(dim, index));
}

torch::Tensor
LBMNeumannBC::nodeEquilibrium(const std::vector<torch::indexing::TensorIndex> & index)
{
  return boundaryEquilibrium(ownedView(_velocity).index(index), ownedView(_rho).index(index));
}

void
LBMNeumannBC::topBoundary()
{
  if (_top_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(1, _shape[1] - 1);

  auto update = faceEquilibrium(1, _shape[1] - 1).index_select(-1, _top_dirs) +
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

  auto update = faceEquilibrium(1, 0).index_select(-1, _bottom_dirs) +
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

  auto update = faceEquilibrium(0, 0).index_select(-1, _left_dirs) +
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

  auto update = faceEquilibrium(0, _shape[0] - 1).index_select(-1, _right_dirs) +
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

  auto update = faceEquilibrium(2, 0).index_select(-1, _front_dirs) +
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

  auto update = faceEquilibrium(2, _shape[2] - 1).index_select(-1, _back_dirs) +
                _f_old_owned.select(2, _shape[2] - 1).index_select(-1, _back_dirs) -
                _feq_owned.select(2, _shape[2] - 1).index_select(-1, _back_dirs);

  u_face.index_copy_(-1, _back_dirs, update);
}

void
LBMNeumannBC::wallBoundary()
{
  // all directions of the marked nodes: gather/scatter whole [K, Q] rows
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_index = nonzeroIndex(ownedView(_binary_mesh) == -1);
    // the media copy is not needed once the boundary entries are indexed
    _binary_mesh = torch::Tensor();
  }

  _u_owned.index_put_(_boundary_index,
                      nodeEquilibrium(_boundary_index) +
                          _f_old_owned.index(_boundary_index) - _feq_owned.index(_boundary_index));
}

void
LBMNeumannBC::regionalBoundary()
{
  // all directions of the marked nodes: gather/scatter whole [K, Q] rows
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_index = nonzeroIndex(ownedView(_binary_mesh) == _region_id);
    // the media copy is not needed once the boundary entries are indexed
    _binary_mesh = torch::Tensor();
  }

  _u_owned.index_put_(_boundary_index,
                      nodeEquilibrium(_boundary_index) +
                          _f_old_owned.index(_boundary_index) - _feq_owned.index(_boundary_index));
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

  LBMBoundaryCondition::computeBuffer(); // Executes the boundary assignments
}
