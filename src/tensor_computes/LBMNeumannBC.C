
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMNeumannBC.h"

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
    // mark 7 (128 in decimal) for regional boundary ownership
    if (isBoundaryOwned(_region_id))
      _boundary_rank |= (1 << 7);
  }
  else if (!isParamValid("region_id") && _lb_problem.isBinaryMedia())
    maskBoundary();
}

void
LBMNeumannBC::computeBoundaryEquilibrium()
{
  const int dim = _domain.getDim();
  torch::Tensor phi_G = _rho + _gradient_value;
  torch::Tensor rho_unsqueezed = phi_G.unsqueeze(3).expand_as(_feq);
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
LBMNeumannBC::topBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    auto opposite = _stencil._op[_stencil._bottom[i]].item<int64_t>();
    _u_owned.index_put_({Slice(), _shape[1] - 1, Slice(), opposite},
                        _feq_boundary.index({Slice(), _shape[1] - 1, Slice(), opposite}) +
                            (_f_old_owned.index({Slice(), _shape[1] - 1, Slice(), opposite}) -
                             ownedView(_feq).index({Slice(), _shape[1] - 1, Slice(), opposite})));
  }
}

void
LBMNeumannBC::bottomBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    _u_owned.index_put_({Slice(), 0, Slice(), _stencil._bottom[i]},
                        _feq_boundary.index({Slice(), 0, Slice(), _stencil._bottom[i]}) +
                            (_f_old_owned.index({Slice(), 0, Slice(), _stencil._bottom[i]}) -
                             ownedView(_feq).index({Slice(), 0, Slice(), _stencil._bottom[i]})));
  }
}

void
LBMNeumannBC::leftBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    _u_owned.index_put_({0, Slice(), Slice(), _stencil._left[i]},
                        _feq_boundary.index({0, Slice(), Slice(), _stencil._left[i]}) +
                            (_f_old_owned.index({0, Slice(), Slice(), _stencil._left[i]}) -
                             ownedView(_feq).index({0, Slice(), Slice(), _stencil._left[i]})));
  }
}

void
LBMNeumannBC::rightBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    auto opposite = _stencil._op[_stencil._left[i]].item<int64_t>();
    _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), opposite},
                        _feq_boundary.index({_shape[0] - 1, Slice(), Slice(), opposite}) +
                            (_f_old_owned.index({_shape[0] - 1, Slice(), Slice(), opposite}) -
                             ownedView(_feq).index({_shape[0] - 1, Slice(), Slice(), opposite})));
  }
}

void
LBMNeumannBC::frontBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    _u_owned.index_put_({Slice(), Slice(), 0, _stencil._front[i]},
                        _feq_boundary.index({Slice(), Slice(), 0, _stencil._front[i]}) +
                            (_f_old_owned.index({Slice(), Slice(), 0, _stencil._front[i]}) -
                             ownedView(_feq).index({Slice(), Slice(), 0, _stencil._front[i]})));
  }
}

void
LBMNeumannBC::backBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    auto opposite = _stencil._op[_stencil._front[i]].item<int64_t>();
    _u_owned.index_put_({Slice(), Slice(), _shape[2] - 1, opposite},
                        _feq_boundary.index({Slice(), Slice(), _shape[2] - 1, opposite}) +
                            (_f_old_owned.index({Slice(), Slice(), _shape[2] - 1, opposite}) -
                             ownedView(_feq).index({Slice(), Slice(), _shape[2] - 1, opposite})));
  }
}

void
LBMNeumannBC::wallBoundary()
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
LBMNeumannBC::regionalBoundary()
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
LBMNeumannBC::computeBuffer()
{
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  computeBoundaryEquilibrium();
  LBMBoundaryCondition::computeBuffer();
}
