/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBounceBack.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMBounceBack);

InputParameters
LBMBounceBack::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMBounceBack object");
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old state distribution function");
  params.addParam<bool>(
      "exclude_corners_x",
      false,
      "Whether or not apply bounceback in the corners of the domain along x axis");
  params.addParam<bool>(
      "exclude_corners_y",
      false,
      "Whether or not apply bounceback in the corners of the domain along y axis");
  params.addParam<bool>(
      "exclude_corners_z",
      false,
      "Whether or not apply bounceback in the corners of the domain along z axis");
  return params;
}

LBMBounceBack::LBMBounceBack(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1, _radius)),
    _exclude_corners_x(getParam<bool>("exclude_corners_x")),
    _exclude_corners_y(getParam<bool>("exclude_corners_y")),
    _exclude_corners_z(getParam<bool>("exclude_corners_z"))
{
  if (_exclude_corners_x)
    _x_indices = torch::arange(1, _shape[0] - 1, MooseTensor::intTensorOptions());
  else
    _x_indices = torch::arange(_shape[0], MooseTensor::intTensorOptions());

  if (_exclude_corners_y)
    _y_indices = torch::arange(1, _shape[1] - 1, MooseTensor::intTensorOptions());
  else
    _y_indices = torch::arange(_shape[1], MooseTensor::intTensorOptions());

  if (_exclude_corners_z)
    _z_indices = torch::arange(1, _shape[2] - 1, MooseTensor::intTensorOptions());
  else
    _z_indices = torch::arange(_shape[2], MooseTensor::intTensorOptions());

  std::vector<torch::Tensor> xyz_mesh = torch::meshgrid({_x_indices, _y_indices, _z_indices});

  torch::Tensor flat_x_indices = xyz_mesh[0].reshape(-1);
  torch::Tensor flat_y_indices = xyz_mesh[1].reshape(-1);
  torch::Tensor flat_z_indices = xyz_mesh[2].reshape(-1);

  _x_indices = flat_x_indices.clone();
  _y_indices = flat_y_indices.clone();
  _z_indices = flat_z_indices.clone();

  // for binary media
  if (_lb_problem.isBinaryMedia())
  {
    const torch::Tensor & binary_mesh = _lb_problem.getBinaryMedia();
    _binary_mesh = binary_mesh.clone();

    // mark 6 (64 in decimal) for wall boundary ownership
    if (isBoundaryOwned(0))
      _boundary_rank |= (1 << 6);

    for (int64_t ic = 1; ic < _stencil._q; ic++)
    {
      int64_t ex = _stencil._ex[ic].item<int64_t>();
      int64_t ey = _stencil._ey[ic].item<int64_t>();
      int64_t ez = _stencil._ez[ic].item<int64_t>();
      torch::Tensor shifted_mesh = torch::roll(binary_mesh, {ex, ey, ez}, {0, 1, 2});
      torch::Tensor adjacent_to_boundary = (shifted_mesh == 0) & (binary_mesh >= 1);
      _binary_mesh.masked_fill_(adjacent_to_boundary, -1);
    }
  }
}

void
LBMBounceBack::backBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u_owned.index_put_(
        {_x_indices, _y_indices, _shape[2] - 1, opposite_dir},
        _f_old_owned.index({_x_indices, _y_indices, _shape[2] - 1, _stencil._front[i]}));
  }
}

void
LBMBounceBack::frontBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u_owned.index_put_({_x_indices, _y_indices, 0, _stencil._front[i]},
                        _f_old_owned.index({_x_indices, _y_indices, 0, opposite_dir}));
  }
}

void
LBMBounceBack::leftBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u_owned.index_put_({0, _y_indices, _z_indices, _stencil._left[i]},
                        _f_old_owned.index({0, _y_indices, _z_indices, opposite_dir}));
  }
}

void
LBMBounceBack::rightBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u_owned.index_put_(
        {_shape[0] - 1, _y_indices, _z_indices, opposite_dir},
        _f_old_owned.index({_shape[0] - 1, _y_indices, _z_indices, _stencil._left[i]}));
  }
}

void
LBMBounceBack::bottomBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u_owned.index_put_({_x_indices, 0, _z_indices, _stencil._bottom[i]},
                        _f_old_owned.index({_x_indices, 0, _z_indices, opposite_dir}));
  }
}

void
LBMBounceBack::topBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u_owned.index_put_(
        {_x_indices, _shape[1] - 1, _z_indices, opposite_dir},
        _f_old_owned.index({_x_indices, _shape[1] - 1, _z_indices, _stencil._bottom[i]}));
  }
}

void
LBMBounceBack::wallBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (_binary_mesh.unsqueeze(-1).expand_as(_u_owned) == -1) & (_u_owned == 0);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  torch::Tensor f_bounce_back = torch::zeros_like(_u_owned);
  for (/* do not use unsigned int */ int ic = 1; ic < _stencil._q; ic++)
  {
    int64_t index = _stencil._op[ic].item<int64_t>();
    auto lattice_slice = _f_old_owned.index({Slice(), Slice(), Slice(), index});
    auto bounce_back_slice = f_bounce_back.index({Slice(), Slice(), Slice(), ic});
    bounce_back_slice.copy_(lattice_slice);
  }
  _u_owned.index_put_({_boundary_mask}, f_bounce_back.index({_boundary_mask}));
}

void
LBMBounceBack::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old == 0)
    return;

  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
