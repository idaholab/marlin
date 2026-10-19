
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMDirichletBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMDirichletBC);

InputParameters
LBMDirichletBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old state distribution function");
  params.addClassDescription("LBMDirichletBC object");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution function");
  params.addRequiredParam<TensorInputBufferName>("velocity", "Fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("rho", "Fluid density");
  params.addParam<Real>("value",
                        "0.0"
                        "Value at the boundary");
  params.addParam<int>("region_id",
                       "0"
                       "Region ID for regional boundary condition");
  return params;
}

LBMDirichletBC::LBMDirichletBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _feq(getInputBuffer("feq", _radius)),
    _rho(getInputBuffer("rho", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _boundary_value(getParam<Real>("value"))
{

  if (isParamValid("region_id") && _lb_problem.isBinaryMedia())
  {
    _region_id = getParam<int>("region_id");
    if (isBoundaryOwned(_region_id))
      _boundary_rank |= (1 << 7);
  }
  else if (!isParamValid("region_id") && _lb_problem.isBinaryMedia())
    maskBoundary();
}

torch::Tensor
LBMDirichletBC::boundaryEquilibrium(const torch::Tensor & velocity, const torch::Tensor & /*rho*/)
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

  // Multiply by the scalar boundary density
  feq_b.mul_(_boundary_value);

  auto shape = velocity.sizes().vec();
  shape.back() = Q;
  return feq_b.view(shape);
}

torch::Tensor
LBMDirichletBC::faceEquilibrium(int64_t dim, int64_t index)
{
  return boundaryEquilibrium(ownedView(_velocity).select(dim, index), torch::Tensor());
}

torch::Tensor
LBMDirichletBC::nodeEquilibrium(const std::vector<torch::indexing::TensorIndex> & index)
{
  return boundaryEquilibrium(ownedView(_velocity).index(index), torch::Tensor());
}

void
LBMDirichletBC::topBoundary()
{
  // select() completely eliminates the loop over Q. It applies to all velocities instantly.
  auto u_face = _u_owned.select(1, _shape[1] - 1);
  auto feq_b_face = faceEquilibrium(1, _shape[1] - 1);
  auto f_old_face = _f_old_owned.select(1, _shape[1] - 1);
  auto feq_face = _feq_owned.select(1, _shape[1] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::bottomBoundary()
{
  auto u_face = _u_owned.select(1, 0);
  auto feq_b_face = faceEquilibrium(1, 0);
  auto f_old_face = _f_old_owned.select(1, 0);
  auto feq_face = _feq_owned.select(1, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::leftBoundary()
{
  auto u_face = _u_owned.select(0, 0);
  auto feq_b_face = faceEquilibrium(0, 0);
  auto f_old_face = _f_old_owned.select(0, 0);
  auto feq_face = _feq_owned.select(0, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::rightBoundary()
{
  auto u_face = _u_owned.select(0, _shape[0] - 1);
  auto feq_b_face = faceEquilibrium(0, _shape[0] - 1);
  auto f_old_face = _f_old_owned.select(0, _shape[0] - 1);
  auto feq_face = _feq_owned.select(0, _shape[0] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::frontBoundary()
{
  auto u_face = _u_owned.select(2, 0);
  auto feq_b_face = faceEquilibrium(2, 0);
  auto f_old_face = _f_old_owned.select(2, 0);
  auto feq_face = _feq_owned.select(2, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::backBoundary()
{
  auto u_face = _u_owned.select(2, _shape[2] - 1);
  auto feq_b_face = faceEquilibrium(2, _shape[2] - 1);
  auto f_old_face = _f_old_owned.select(2, _shape[2] - 1);
  auto feq_face = _feq_owned.select(2, _shape[2] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::wallBoundary()
{
  // all directions of the marked nodes: gather/scatter whole [K, Q] rows
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_index = nonzeroIndex(ownedView(_binary_mesh) == -1);
    // the media copy is not needed once the boundary entries are indexed
    _binary_mesh = torch::Tensor();
  }

  // Computed in a single vectorized pass
  _u_owned.index_put_(_boundary_index,
                      nodeEquilibrium(_boundary_index) +
                          _f_old_owned.index(_boundary_index) - _feq_owned.index(_boundary_index));
}

void
LBMDirichletBC::regionalBoundary()
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
LBMDirichletBC::computeBuffer()
{
  // Prepare f_old for active domain
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  // Cache owned views for the step to avoid repeated allocations in the loops
  _feq_owned = ownedView(_feq);

  LBMBoundaryCondition::computeBuffer(); // Executes the boundary assignments
}
