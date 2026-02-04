
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMDirichletBC.h"

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
    // mark 7 (128 in decimal) for regional boundary ownership
    if (isBoundaryOwned(_region_id))
      _boundary_rank |= (1 << 7);
  }
  else if (!isParamValid("region_id") && _lb_problem.isBinaryMedia())
    maskBoundary();
}

void
LBMDirichletBC::computeBoundaryEquilibrium()
{
  const int dim = _domain.getDim();
  auto rho_unsqueezed = torch::full_like(_feq, _boundary_value);
  torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  torch::Tensor uz;

  switch (dim)
  {
    case 3:
      uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    case 2:
      uz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }

  torch::Tensor second_order;
  torch::Tensor third_order;
  {
    auto edotu = _ex * ux + _ey * uy + _ez * uz;
    auto edotu_sqr = edotu * edotu;
    auto usqr = ux * ux + uy * uy + uz * uz;
    second_order = edotu / _lb_problem._cs2 + 0.5 * edotu_sqr / _lb_problem._cs4;
    third_order = 0.5 * usqr / _lb_problem._cs2;
  }
  auto feq_boundary = _w * rho_unsqueezed * (1.0 + second_order - third_order);
  _feq_boundary = ownedView(feq_boundary);
}

void
LBMDirichletBC::topBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({Slice(), _shape[1] - 1, Slice(), i},
                        _feq_boundary.index({Slice(), _shape[1] - 1, Slice(), i}) +
                            (_f_old_owned.index({Slice(), _shape[1] - 1, Slice(), i}) -
                             ownedView(_feq).index({Slice(), _shape[1] - 1, Slice(), i})));
  }
}

void
LBMDirichletBC::bottomBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({Slice(), 0, Slice(), i},
                        _feq_boundary.index({Slice(), 0, Slice(), i}) +
                            (_f_old_owned.index({Slice(), 0, Slice(), i}) -
                             ownedView(_feq).index({Slice(), 0, Slice(), i})));
  }
}

void
LBMDirichletBC::leftBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({0, Slice(), Slice(), i},
                        _feq_boundary.index({0, Slice(), Slice(), i}) +
                            (_f_old_owned.index({0, Slice(), Slice(), i}) -
                             ownedView(_feq).index({0, Slice(), Slice(), i})));
  }
}

void
LBMDirichletBC::rightBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), i},
                        _feq_boundary.index({_shape[0] - 1, Slice(), Slice(), i}) +
                            (_f_old_owned.index({_shape[0] - 1, Slice(), Slice(), i}) -
                             ownedView(_feq).index({_shape[0] - 1, Slice(), Slice(), i})));
  }
}

void
LBMDirichletBC::frontBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({Slice(), Slice(), 0, i},
                        _feq_boundary.index({Slice(), Slice(), 0, i}) +
                            (_f_old_owned.index({Slice(), Slice(), 0, i}) -
                             ownedView(_feq).index({Slice(), Slice(), 0, i})));
  }
}

void
LBMDirichletBC::backBoundary()
{
  for (int64_t i = 0; i < _stencil._q; i++)
  {
    _u_owned.index_put_({Slice(), Slice(), _shape[2] - 1, i},
                        _feq_boundary.index({Slice(), Slice(), _shape[2] - 1, i}) +
                            (_f_old_owned.index({Slice(), Slice(), _shape[2] - 1, i}) -
                             ownedView(_feq).index({Slice(), Slice(), _shape[2] - 1, i})));
  }
}

void
LBMDirichletBC::wallBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == -1);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }
  _u_owned.index_put_(
      {_boundary_mask},
      _feq_boundary.index({_boundary_mask}) +
          (_f_old_owned.index({_boundary_mask}) - ownedView(_feq).index({_boundary_mask})));
}

void
LBMDirichletBC::regionalBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == _region_id);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }
  _u_owned.index_put_(
      {_boundary_mask},
      _feq_boundary.index({_boundary_mask}) +
          (_f_old_owned.index({_boundary_mask}) - ownedView(_feq).index({_boundary_mask})));
}

void
LBMDirichletBC::computeBuffer()
{
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  computeBoundaryEquilibrium();
  LBMBoundaryCondition::computeBuffer();
}
