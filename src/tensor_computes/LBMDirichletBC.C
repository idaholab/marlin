
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
  _feq_boundary = torch::zeros_like(_feq);

  if (isParamValid("region_id") && _lb_problem.isBinaryMedia())
  {
    _region_id = getParam<int>("region_id");
    if (isBoundaryOwned(_region_id))
      _boundary_rank |= (1 << 7);
  }
  else if (!isParamValid("region_id") && _lb_problem.isBinaryMedia())
    maskBoundary();
}

void
LBMDirichletBC::computeBoundaryEquilibrium()
{
  const unsigned int dim = _domain.getDim();
  auto vel_owned = ownedView(_velocity);

  const int64_t N = vel_owned.numel() / vel_owned.size(-1);
  auto vel_flat = vel_owned.slice(-1, 0, dim).reshape({N, dim});

  // macroscopic variables
  auto usqr = vel_flat.square().sum(-1, /*keepdim=*/true);
  auto edotu = torch::mm(vel_flat, _e_mat.t());

  auto edotu_spatial = edotu.reshape_as(_feq_boundary_owned);

  auto usqr_shape = _feq_boundary_owned.sizes().vec();
  usqr_shape.back() = 1;
  auto usqr_spatial = usqr.reshape(usqr_shape);

  // In-place polynomial: feq_b = w * rho_bc * (1 + edotu/cs2 + edotu^2/2cs4 - usqr/2cs2)
  _feq_boundary_owned.copy_(edotu_spatial).square_().div_(2.0 * _lb_problem._cs4);
  _feq_boundary_owned.add_(edotu_spatial, 1.0 / _lb_problem._cs2);
  _feq_boundary_owned.sub_(usqr_spatial, 1.0 / (2.0 * _lb_problem._cs2));
  _feq_boundary_owned.add_(1.0);
  _feq_boundary_owned.mul_(_w);

  // Multiply by the scalar boundary density
  _feq_boundary_owned.mul_(_boundary_value);
}

void
LBMDirichletBC::topBoundary()
{
  // select() completely eliminates the loop over Q. It applies to all velocities instantly.
  auto u_face = _u_owned.select(1, _shape[1] - 1);
  auto feq_b_face = _feq_boundary_owned.select(1, _shape[1] - 1);
  auto f_old_face = _f_old_owned.select(1, _shape[1] - 1);
  auto feq_face = _feq_owned.select(1, _shape[1] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::bottomBoundary()
{
  auto u_face = _u_owned.select(1, 0);
  auto feq_b_face = _feq_boundary_owned.select(1, 0);
  auto f_old_face = _f_old_owned.select(1, 0);
  auto feq_face = _feq_owned.select(1, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::leftBoundary()
{
  auto u_face = _u_owned.select(0, 0);
  auto feq_b_face = _feq_boundary_owned.select(0, 0);
  auto f_old_face = _f_old_owned.select(0, 0);
  auto feq_face = _feq_owned.select(0, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::rightBoundary()
{
  auto u_face = _u_owned.select(0, _shape[0] - 1);
  auto feq_b_face = _feq_boundary_owned.select(0, _shape[0] - 1);
  auto f_old_face = _f_old_owned.select(0, _shape[0] - 1);
  auto feq_face = _feq_owned.select(0, _shape[0] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::frontBoundary()
{
  auto u_face = _u_owned.select(2, 0);
  auto feq_b_face = _feq_boundary_owned.select(2, 0);
  auto f_old_face = _f_old_owned.select(2, 0);
  auto feq_face = _feq_owned.select(2, 0);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::backBoundary()
{
  auto u_face = _u_owned.select(2, _shape[2] - 1);
  auto feq_b_face = _feq_boundary_owned.select(2, _shape[2] - 1);
  auto f_old_face = _f_old_owned.select(2, _shape[2] - 1);
  auto feq_face = _feq_owned.select(2, _shape[2] - 1);

  u_face.copy_(feq_b_face + f_old_face - feq_face);
}

void
LBMDirichletBC::wallBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == -1);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  // Computed in a single vectorized pass
  _u_owned.index_put_({_boundary_mask},
                      _feq_boundary_owned.index({_boundary_mask}) +
                          _f_old_owned.index({_boundary_mask}) -
                          _feq_owned.index({_boundary_mask}));
}

void
LBMDirichletBC::regionalBoundary()
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
LBMDirichletBC::computeBuffer()
{
  // Prepare f_old for active domain
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  // Cache owned views for the step to avoid repeated allocations in the loops
  _feq_owned = ownedView(_feq);
  _feq_boundary_owned = ownedView(_feq_boundary);

  computeBoundaryEquilibrium();
  LBMBoundaryCondition::computeBuffer(); // Executes the boundary assignments
}
