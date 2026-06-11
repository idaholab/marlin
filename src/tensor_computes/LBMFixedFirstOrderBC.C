/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMFixedFirstOrderBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

#include <cstdlib>

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMFixedFirstOrderBC);

InputParameters
LBMFixedFirstOrderBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMFixedFirstOrderBC object");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<std::string>("value", "Fixed input velocity");
  params.addParam<bool>("perturb", false, "Whether to perturb first order moment at the boundary");
  return params;
}

LBMFixedFirstOrderBC::LBMFixedFirstOrderBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f(getInputBufferByName(getParam<TensorInputBufferName>("f"), _radius)),
    _value(_lb_problem.getConstant<Real>(getParam<std::string>("value"))),
    _perturb(getParam<bool>("perturb"))
{
}

void
LBMFixedFirstOrderBC::frontBoundary()
{
  if (_domain.getDim() == 2)
    mooseError("There is no front boundary in 2 dimensions.");
  else
    mooseError("Front boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain.");
}

void
LBMFixedFirstOrderBC::backBoundary()
{
  if (_domain.getDim() == 2)
    mooseError("There is no back boundary in 2 dimensions.");
  else
    mooseError("Back boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain.");
}

void
LBMFixedFirstOrderBC::leftBoundaryD2Q9()
{
  if (_u_x_perturbed.numel() == 0)
  {
    auto rank = _domain.comm().rank();
    std::array<int64_t, 3> begin, end;
    _domain.getLocalBounds(rank, begin, end);
    auto n_global = _domain.getGridSize();

    _u_x_perturbed = torch::empty({end[1] - begin[1], 1}, MooseTensor::floatTensorOptions());

    if (_perturb)
    {
      Real deltaU = 1.0e-6 * _value;
      torch::Tensor y_coords =
          torch::arange(begin[1], end[1], MooseTensor::floatTensorOptions()).unsqueeze(1) /
          n_global[1];
      _u_x_perturbed = _value + deltaU * torch::sin(y_coords * 2.0 * M_PI);
    }
    else
    {
      _u_x_perturbed.fill_(_value);
    }
  }

  auto f_face = _f_owned.select(0, 0);
  auto u_face = _u_owned.select(0, 0);

  auto f0 = f_face.select(-1, 0);
  auto f2 = f_face.select(-1, 2);
  auto f4 = f_face.select(-1, 4);
  auto f3 = f_face.select(-1, 3);
  auto f6 = f_face.select(-1, 6);
  auto f7 = f_face.select(-1, 7);

  auto density = torch::add(f3, f6).add_(f7).mul_(2.0).add_(f0).add_(f2).add_(f4);

  auto u_x_safe =
      _u_x_perturbed.dim() < f_face.dim() - 1 ? _u_x_perturbed.unsqueeze(-1) : _u_x_perturbed;
  density.div_(1.0 - u_x_safe);

  int left_0 = _stencil._left[0].item<int>();
  int opp_0 = _stencil._op[left_0].item<int>();

  u_face.select(-1, left_0).copy_(f_face.select(-1, opp_0)).addcmul_(density, u_x_safe, 2.0 / 3.0);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int opp_i = _stencil._op[left_i].item<int>();
    double ey = _stencil._ey[left_i].item<double>();

    auto update =
        f_face.select(-1, opp_i) - 0.5 * ey * (f2 - f4) + (1.0 / 6.0) * density * u_x_safe;
    u_face.select(-1, left_i).copy_(update);
  }
}

void
LBMFixedFirstOrderBC::leftBoundary()
{
  if (_stencil._q == 9)
  {
    leftBoundaryD2Q9();
    return;
  }

  auto f_face = _f_owned.select(0, 0);
  auto u_face = _u_owned.select(0, 0);

  auto safe_neutral_x = (-_stencil._neutral_x).remainder(_stencil._q);
  auto f_neutral = f_face.index_select(-1, safe_neutral_x).sum(-1);
  auto f_right = f_face.index_select(-1, _stencil._right).sum(-1);

  auto density = (f_neutral + 2.0 * f_right).div_(1.0 - _value);

  int left_0 = _stencil._left[0].item<int>();
  int right_0 = _stencil._right[0].item<int>();
  double w_left_0 = _stencil._weights[left_0].item<double>();

  u_face.select(-1, left_0)
      .copy_(f_face.select(-1, right_0))
      .add_(density, 2.0 * w_left_0 / _lb_problem._cs2 * _value);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int right_i = _stencil._right[i].item<int>();
    double w_left_i = _stencil._weights[left_i].item<double>();

    u_face.select(-1, left_i)
        .copy_(f_face.select(-1, right_i))
        .add_(density, 2.0 * w_left_i / _lb_problem._cs2 * _value);
  }
}

void
LBMFixedFirstOrderBC::rightBoundaryD2Q9()
{
  auto f_face = _f_owned.select(0, _shape[0] - 1);
  auto u_face = _u_owned.select(0, _shape[0] - 1);

  auto f0 = f_face.select(-1, 0);
  auto f2 = f_face.select(-1, 2);
  auto f4 = f_face.select(-1, 4);
  auto f1 = f_face.select(-1, 1);
  auto f5 = f_face.select(-1, 5);
  auto f8 = f_face.select(-1, 8);

  auto density = torch::add(f1, f5).add_(f8).mul_(2.0).add_(f0).add_(f2).add_(f4);
  density.div_(1.0 + _value);

  int left_0 = _stencil._left[0].item<int>();
  int opp_0 = _stencil._op[left_0].item<int>();

  u_face.select(-1, opp_0).copy_(f_face.select(-1, left_0)).add_(density, -2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int opp_i = _stencil._op[left_i].item<int>();
    double ey_opp = _stencil._ey[opp_i].item<double>();

    auto update =
        f_face.select(-1, left_i) + 0.5 * ey_opp * (f4 - f2) - (1.0 / 6.0) * density * _value;
    u_face.select(-1, opp_i).copy_(update);
  }
}

void
LBMFixedFirstOrderBC::rightBoundary()
{
  if (_stencil._q == 9)
  {
    rightBoundaryD2Q9();
    return;
  }

  auto f_face = _f_owned.select(0, _shape[0] - 1);
  auto u_face = _u_owned.select(0, _shape[0] - 1);

  auto safe_neutral_x = (-_stencil._neutral_x).remainder(_stencil._q);
  auto f_neutral = f_face.index_select(-1, safe_neutral_x).sum(-1);
  auto f_left = f_face.index_select(-1, _stencil._left).sum(-1);

  auto density = (f_neutral + 2.0 * f_left).div_(1.0 + _value);

  int right_0 = _stencil._right[0].item<int>();
  int left_0 = _stencil._left[0].item<int>();
  double w_right_0 = _stencil._weights[right_0].item<double>();

  u_face.select(-1, right_0)
      .copy_(f_face.select(-1, left_0))
      .add_(density, -2.0 * w_right_0 / _lb_problem._cs2 * _value);

  for (unsigned int i = 1; i < _stencil._right.size(0); i++)
  {
    int right_i = _stencil._right[i].item<int>();
    int left_i = _stencil._left[i].item<int>();
    double w_right_i = _stencil._weights[right_i].item<double>();

    u_face.select(-1, right_i)
        .copy_(f_face.select(-1, left_i))
        .add_(density, -2.0 * w_right_i / _lb_problem._cs2 * _value);
  }
}

void
LBMFixedFirstOrderBC::bottomBoundaryD2Q9()
{
  auto f_face = _f_owned.select(1, 0);
  auto u_face = _u_owned.select(1, 0);

  auto f0 = f_face.select(-1, 0);
  auto f1 = f_face.select(-1, 1);
  auto f3 = f_face.select(-1, 3);
  auto f4 = f_face.select(-1, 4);
  auto f7 = f_face.select(-1, 7);
  auto f8 = f_face.select(-1, 8);

  auto density = torch::add(f4, f7).add_(f8).mul_(2.0).add_(f0).add_(f1).add_(f3);
  density.div_(1.0 - _value);

  int bot_0 = _stencil._bottom[0].item<int>();
  int opp_0 = _stencil._op[bot_0].item<int>();

  u_face.select(-1, bot_0).copy_(f_face.select(-1, opp_0)).add_(density, 2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    int bot_i = _stencil._bottom[i].item<int>();
    int opp_i = _stencil._op[bot_i].item<int>();
    double ex = _stencil._ex[bot_i].item<double>();

    auto update = f_face.select(-1, opp_i) - 0.5 * ex * (f1 - f3) + (1.0 / 6.0) * density * _value;
    u_face.select(-1, bot_i).copy_(update);
  }
}

void
LBMFixedFirstOrderBC::bottomBoundary()
{
  if (_stencil._q == 9)
    bottomBoundaryD2Q9();
  else
    mooseError("Bottom boundary is not implemented, but it can be replaced by another boundary by "
               "rotating the domain.");
}

void
LBMFixedFirstOrderBC::topBoundaryD2Q9()
{
  auto f_face = _f_owned.select(1, _shape[1] - 1);
  auto u_face = _u_owned.select(1, _shape[1] - 1);

  auto f0 = f_face.select(-1, 0);
  auto f1 = f_face.select(-1, 1);
  auto f3 = f_face.select(-1, 3);
  auto f2 = f_face.select(-1, 2);
  auto f5 = f_face.select(-1, 5);
  auto f6 = f_face.select(-1, 6);

  auto density = torch::add(f2, f5).add_(f6).mul_(2.0).add_(f0).add_(f1).add_(f3);
  density.div_(1.0 + _value);

  int bot_0 = _stencil._bottom[0].item<int>();
  int opp_0 = _stencil._op[bot_0].item<int>();

  u_face.select(-1, opp_0).copy_(f_face.select(-1, bot_0)).add_(density, -2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    int bot_i = _stencil._bottom[i].item<int>();
    int opp_i = _stencil._op[bot_i].item<int>();
    double ex_opp = _stencil._ex[opp_i].item<double>();

    auto update =
        f_face.select(-1, bot_i) + 0.5 * ex_opp * (f3 - f1) - (1.0 / 6.0) * density * _value;
    u_face.select(-1, opp_i).copy_(update);
  }
}

void
LBMFixedFirstOrderBC::topBoundary()
{
  if (_stencil._q == 9)
    topBoundaryD2Q9();
  else
    mooseError("Top boundary is not implemented, but it can be replaced by another boundary by "
               "rotating the domain.");
}

void
LBMFixedFirstOrderBC::computeBuffer()
{
  _f_owned = _f;
  for (unsigned int d = 0; d < _dim; d++)
    _f_owned = _f_owned.narrow(d, _radius, _shape[d]);

  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
