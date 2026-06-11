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
  _x_start = _exclude_corners_x ? 1 : 0;
  _x_end = _exclude_corners_x ? _shape[0] - 1 : _shape[0];

  _y_start = _exclude_corners_y ? 1 : 0;
  _y_end = _exclude_corners_y ? _shape[1] - 1 : _shape[1];

  _z_start = _exclude_corners_z ? 1 : 0;
  _z_end = _exclude_corners_z ? _shape[2] - 1 : _shape[2];

  auto cache_dirs =
      [&](const torch::Tensor & dirs, torch::Tensor & out_dirs, torch::Tensor & out_opps)
  {
    if (dirs.size(0) > 0)
    {
      out_dirs = dirs.to(torch::kLong);
      out_opps = _stencil._op.index_select(0, out_dirs).to(torch::kLong);
    }
  };

  // reuse these arrays with flipped assignments to maintain the original physics logic.
  cache_dirs(_stencil._left, _left_dirs, _left_opp_dirs);
  cache_dirs(_stencil._bottom, _bottom_dirs, _bottom_opp_dirs);
  cache_dirs(_stencil._front, _front_dirs, _front_opp_dirs);

  if (_lb_problem.isBinaryMedia())
    maskBoundary();
}

void
LBMBounceBack::leftBoundary()
{
  if (_left_dirs.numel() == 0)
    return;
  // select(0, 0) grabs the X=0 plane. slice() bounds the Y and Z corners.
  auto u_face = _u_owned.select(0, 0).slice(0, _y_start, _y_end).slice(1, _z_start, _z_end);
  auto f_old_face = _f_old_owned.select(0, 0).slice(0, _y_start, _y_end).slice(1, _z_start, _z_end);

  // u[left] = f_old[right] for all Q simultaneously
  u_face.index_copy_(-1, _left_dirs, f_old_face.index_select(-1, _left_opp_dirs));
}

void
LBMBounceBack::rightBoundary()
{
  if (_left_dirs.numel() == 0)
    return;
  auto u_face =
      _u_owned.select(0, _shape[0] - 1).slice(0, _y_start, _y_end).slice(1, _z_start, _z_end);
  auto f_old_face =
      _f_old_owned.select(0, _shape[0] - 1).slice(0, _y_start, _y_end).slice(1, _z_start, _z_end);

  // u[right] = f_old[left] for all Q simultaneously
  u_face.index_copy_(-1, _left_opp_dirs, f_old_face.index_select(-1, _left_dirs));
}

void
LBMBounceBack::bottomBoundary()
{
  if (_bottom_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(1, 0).slice(0, _x_start, _x_end).slice(1, _z_start, _z_end);
  auto f_old_face = _f_old_owned.select(1, 0).slice(0, _x_start, _x_end).slice(1, _z_start, _z_end);

  u_face.index_copy_(-1, _bottom_dirs, f_old_face.index_select(-1, _bottom_opp_dirs));
}

void
LBMBounceBack::topBoundary()
{
  if (_bottom_dirs.numel() == 0)
    return;
  auto u_face =
      _u_owned.select(1, _shape[1] - 1).slice(0, _x_start, _x_end).slice(1, _z_start, _z_end);
  auto f_old_face =
      _f_old_owned.select(1, _shape[1] - 1).slice(0, _x_start, _x_end).slice(1, _z_start, _z_end);

  u_face.index_copy_(-1, _bottom_opp_dirs, f_old_face.index_select(-1, _bottom_dirs));
}

void
LBMBounceBack::frontBoundary()
{
  if (_front_dirs.numel() == 0)
    return;
  auto u_face = _u_owned.select(2, 0).slice(0, _x_start, _x_end).slice(1, _y_start, _y_end);
  auto f_old_face = _f_old_owned.select(2, 0).slice(0, _x_start, _x_end).slice(1, _y_start, _y_end);

  u_face.index_copy_(-1, _front_dirs, f_old_face.index_select(-1, _front_opp_dirs));
}

void
LBMBounceBack::backBoundary()
{
  if (_front_dirs.numel() == 0)
    return;
  auto u_face =
      _u_owned.select(2, _shape[2] - 1).slice(0, _x_start, _x_end).slice(1, _y_start, _y_end);
  auto f_old_face =
      _f_old_owned.select(2, _shape[2] - 1).slice(0, _x_start, _x_end).slice(1, _y_start, _y_end);

  u_face.index_copy_(-1, _front_opp_dirs, f_old_face.index_select(-1, _front_dirs));
}

void
LBMBounceBack::wallBoundary()
{
  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask =
        (ownedView(_binary_mesh).unsqueeze(-1).expand_as(_u_owned) == -1) & (_u_owned == 0);
    _boundary_mask = _boundary_mask.to(torch::kBool);

    _op_indices = _stencil._op.to(torch::kLong);
  }

  // Gather all opposite directions simultaneously
  auto f_bounce_back = torch::index_select(_f_old_owned, 3, _op_indices);

  // Boolean masked assignment
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
