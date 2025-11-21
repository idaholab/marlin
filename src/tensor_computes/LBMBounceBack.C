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
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _exclude_corners_x(getParam<bool>("exclude_corners_x")),
    _exclude_corners_y(getParam<bool>("exclude_corners_y")),
    _exclude_corners_z(getParam<bool>("exclude_corners_z"))
{
  if (_exclude_corners_x)
    _x_indices = torch::arange(1, _grid_size[0] - 1, MooseTensor::intTensorOptions());
  else
    _x_indices = torch::arange(_grid_size[0], MooseTensor::intTensorOptions());

  if (_exclude_corners_y)
    _y_indices = torch::arange(1, _grid_size[1] - 1, MooseTensor::intTensorOptions());
  else
    _y_indices = torch::arange(_grid_size[1], MooseTensor::intTensorOptions());

  if (_exclude_corners_z)
    _z_indices = torch::arange(1, _grid_size[2] - 1, MooseTensor::intTensorOptions());
  else
    _z_indices = torch::arange(_grid_size[2], MooseTensor::intTensorOptions());

  std::vector<torch::Tensor> xyz_mesh = torch::meshgrid({_x_indices, _y_indices, _z_indices});

  torch::Tensor flat_x_indices = xyz_mesh[0].reshape(-1);
  torch::Tensor flat_y_indices = xyz_mesh[1].reshape(-1);
  torch::Tensor flat_z_indices = xyz_mesh[2].reshape(-1);

  _x_indices = flat_x_indices.clone();
  _y_indices = flat_y_indices.clone();
  _z_indices = flat_z_indices.clone();

  // for 3D, binary media
  if (_lb_problem.isBinaryMedia())
  {
    const torch::Tensor & binary_mesh = _lb_problem.getBinaryMedia();
    _binary_mesh = binary_mesh.clone();

    for (int64_t ic = 1; ic < _stencil._q; ic++)
    {
      int64_t ex = _stencil._ex[ic].item<int64_t>();
      int64_t ey = _stencil._ey[ic].item<int64_t>();
      int64_t ez = _stencil._ez[ic].item<int64_t>();
      torch::Tensor shifted_mesh = torch::roll(binary_mesh, {ex, ey, ez}, {0, 1, 2});
      torch::Tensor adjacent_to_boundary = (shifted_mesh == 0) & (binary_mesh > 1);
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
    _u.index_put_({_x_indices, _y_indices, _grid_size[2] - 1, opposite_dir},
                  _f_old[0].index({_x_indices, _y_indices, _grid_size[2] - 1, _stencil._front[i]}));
  }
}

void
LBMBounceBack::frontBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({_x_indices, _y_indices, 0, _stencil._front[i]},
                  _f_old[0].index({_x_indices, _y_indices, 0, opposite_dir}));
  }
}

void
LBMBounceBack::leftBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({0, _y_indices, _z_indices, _stencil._left[i]},
                  _f_old[0].index({0, _y_indices, _z_indices, opposite_dir}));
  }
}

void
LBMBounceBack::rightBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_grid_size[0] - 1, _y_indices, _z_indices, opposite_dir},
                  _f_old[0].index({_grid_size[0] - 1, _y_indices, _z_indices, _stencil._left[i]}));
  }
}

void
LBMBounceBack::bottomBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({_x_indices, 0, _z_indices, _stencil._bottom[i]},
                  _f_old[0].index({_x_indices, 0, _z_indices, opposite_dir}));
  }
}

void
LBMBounceBack::topBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_(
        {_x_indices, _grid_size[1] - 1, _z_indices, opposite_dir},
        _f_old[0].index({_x_indices, _grid_size[1] - 1, _z_indices, _stencil._bottom[i]}));
  }
}

void
LBMBounceBack::wallBoundary()
{
  if (_domain.getDim() == 3)
    wallBoundary3D(); // temporary solution to generalization problem
  else
    // bounce-back
    _u.index_put_(
        {_boundary_indices.index({Slice(), 0}),
         _boundary_indices.index({Slice(), 1}),
         _boundary_indices.index({Slice(), 2}),
         _boundary_indices.index({Slice(), 3})},

        _f_old[0].index({_boundary_indices.index({Slice(), 0}),
                         _boundary_indices.index({Slice(), 1}),
                         _boundary_indices.index({Slice(), 2}),
                         _stencil._op.index_select(0, _boundary_indices.index({Slice(), 3}))}));
}

void
LBMBounceBack::wallBoundary3D()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (_binary_mesh.unsqueeze(-1).expand_as(_u) == -1) & (_u == 0);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  torch::Tensor f_bounce_back = torch::zeros_like(_u);

  for (/* do not use unsigned int */ int ic = 1; ic < _stencil._q; ic++)
  {
    int64_t index = _stencil._op[ic].item<int64_t>();
    auto lattice_slice = _f_old[0].index({Slice(), Slice(), Slice(), index});
    auto bounce_back_slice = f_bounce_back.index({Slice(), Slice(), Slice(), ic});
    bounce_back_slice.copy_(lattice_slice);
  }

  _u.index_put_({_boundary_mask}, f_bounce_back.index({_boundary_mask}));
}

void
LBMBounceBack::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
  {
    // do not overwrite previous
    _u = _u.clone();

    switch (_boundary)
    {
      case Boundary::top:
        topBoundary();
        break;
      case Boundary::bottom:
        bottomBoundary();
        break;
      case Boundary::left:
        leftBoundary();
        break;
      case Boundary::right:
        rightBoundary();
        break;
      case Boundary::front:
        frontBoundary();
        break;
      case Boundary::back:
        backBoundary();
        break;
      case Boundary::wall:
        wallBoundary3D();
        break;
      default:
        mooseError("Undefined boundary names");
    }
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
