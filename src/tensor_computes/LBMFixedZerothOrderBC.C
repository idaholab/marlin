/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMFixedZerothOrderBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMFixedZerothOrderBC);

InputParameters
LBMFixedZerothOrderBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMFixedZerothOrderBC object");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<std::string>("value", "Fixed input value");
  return params;
}

LBMFixedZerothOrderBC::LBMFixedZerothOrderBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f(getInputBufferByName(getParam<TensorInputBufferName>("f"), _radius)),
    _value(_lb_problem.getConstant<Real>(getParam<std::string>("value")))
{
}

void
LBMFixedZerothOrderBC::frontBoundary()
{
  if (_domain.getDim() == 2)
    mooseError("There is no front boundary in 2 dimensions.");
  else
    mooseError("Front boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain.");
}

void
LBMFixedZerothOrderBC::backBoundary()
{
  if (_domain.getDim() == 2)
    mooseError("There is no back boundary in 2 dimensions.");
  else
    mooseError("Back boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain.");
}

void
LBMFixedZerothOrderBC::leftBoundaryD2Q9()
{
  auto f_face = _f_owned.select(0, 0);
  auto u_face = _u_owned.select(0, 0);

  auto f0 = f_face.select(-1, 0);
  auto f2 = f_face.select(-1, 2);
  auto f4 = f_face.select(-1, 4);
  auto f3 = f_face.select(-1, 3);
  auto f6 = f_face.select(-1, 6);
  auto f7 = f_face.select(-1, 7);

  auto velocity = torch::add(f3, f6).add_(f7).mul_(2.0).add_(f0).add_(f2).add_(f4);
  velocity.div_(-_value).add_(1.0);

  int left_0 = _stencil._left[0].item<int>();
  int opp_0 = _stencil._op[left_0].item<int>();

  u_face.select(-1, left_0).copy_(f_face.select(-1, opp_0)).add_(velocity, 2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int opp_i = _stencil._op[left_i].item<int>();
    double ey = _stencil._ey[left_i].item<double>();

    auto update = f_face.select(-1, opp_i) - 0.5 * ey * (f2 - f4) + (1.0 / 6.0) * _value * velocity;
    u_face.select(-1, left_i).copy_(update);
  }
}

void
LBMFixedZerothOrderBC::leftBoundary()
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

  auto velocity = (f_neutral + 2.0 * f_right).div_(-_value).add_(1.0);

  int left_0 = _stencil._left[0].item<int>();
  int right_0 = _stencil._right[0].item<int>();
  double w_left_0 = _stencil._weights[left_0].item<double>();

  u_face.select(-1, left_0)
      .copy_(f_face.select(-1, right_0))
      .add_(velocity, 2.0 * w_left_0 / _lb_problem._cs2 * _value);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int right_i = _stencil._right[i].item<int>();
    double w_left_i = _stencil._weights[left_i].item<double>();

    u_face.select(-1, left_i)
        .copy_(f_face.select(-1, right_i))
        .add_(velocity, 2.0 * w_left_i / _lb_problem._cs2 * _value);
  }
}

void
LBMFixedZerothOrderBC::rightBoundaryD2Q9()
{
  auto f_face = _f_owned.select(0, _shape[0] - 1);
  auto u_face = _u_owned.select(0, _shape[0] - 1);

  auto f0 = f_face.select(-1, 0);
  auto f2 = f_face.select(-1, 2);
  auto f4 = f_face.select(-1, 4);
  auto f1 = f_face.select(-1, 1);
  auto f5 = f_face.select(-1, 5);
  auto f8 = f_face.select(-1, 8);

  auto velocity = torch::add(f1, f5).add_(f8).mul_(2.0).add_(f0).add_(f2).add_(f4);
  velocity.div_(_value).sub_(1.0);

  int left_0 = _stencil._left[0].item<int>();
  int opp_0 = _stencil._op[left_0].item<int>();

  u_face.select(-1, opp_0).copy_(f_face.select(-1, left_0)).add_(velocity, -2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    int left_i = _stencil._left[i].item<int>();
    int opp_i = _stencil._op[left_i].item<int>();
    double ey_opp = _stencil._ey[opp_i].item<double>();

    auto update =
        f_face.select(-1, left_i) + 0.5 * ey_opp * (f4 - f2) - (1.0 / 6.0) * _value * velocity;
    u_face.select(-1, opp_i).copy_(update);
  }
}

void
LBMFixedZerothOrderBC::rightBoundary()
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

  auto velocity = (f_neutral + 2.0 * f_left).div_(_value).sub_(1.0);

  int right_0 = _stencil._right[0].item<int>();
  int left_0 = _stencil._left[0].item<int>();
  double w_right_0 = _stencil._weights[right_0].item<double>();

  u_face.select(-1, right_0)
      .copy_(f_face.select(-1, left_0))
      .add_(velocity, -2.0 * w_right_0 / _lb_problem._cs2 * _value);

  for (unsigned int i = 1; i < _stencil._right.size(0); i++)
  {
    int right_i = _stencil._right[i].item<int>();
    int left_i = _stencil._left[i].item<int>();
    double w_right_i = _stencil._weights[right_i].item<double>();

    u_face.select(-1, right_i)
        .copy_(f_face.select(-1, left_i))
        .add_(velocity, -2.0 * w_right_i / _lb_problem._cs2 * _value);
  }
}

void
LBMFixedZerothOrderBC::bottomBoundaryD2Q9()
{
  auto f_face = _f_owned.select(1, 0);
  auto u_face = _u_owned.select(1, 0);

  auto f0 = f_face.select(-1, 0);
  auto f1 = f_face.select(-1, 1);
  auto f3 = f_face.select(-1, 3);
  auto f4 = f_face.select(-1, 4);
  auto f7 = f_face.select(-1, 7);
  auto f8 = f_face.select(-1, 8);

  auto velocity = torch::add(f4, f7).add_(f8).mul_(2.0).add_(f0).add_(f1).add_(f3);
  velocity.div_(-_value).add_(1.0);

  int bot_0 = _stencil._bottom[0].item<int>();
  int opp_0 = _stencil._op[bot_0].item<int>();

  u_face.select(-1, bot_0).copy_(f_face.select(-1, opp_0)).add_(velocity, 2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    int bot_i = _stencil._bottom[i].item<int>();
    int opp_i = _stencil._op[bot_i].item<int>();
    double ex = _stencil._ex[bot_i].item<double>();

    auto update = f_face.select(-1, opp_i) - 0.5 * ex * (f1 - f3) + (1.0 / 6.0) * _value * velocity;
    u_face.select(-1, bot_i).copy_(update);
  }
}

void
LBMFixedZerothOrderBC::bottomBoundary()
{
  if (_stencil._q == 9)
    bottomBoundaryD2Q9();
  else
    mooseError("Bottom boundary is not implemented, but it can be replaced by any other boundary "
               "by rotating the domain");
}

void
LBMFixedZerothOrderBC::topBoundaryD2Q9()
{
  auto f_face = _f_owned.select(1, _shape[1] - 1);
  auto u_face = _u_owned.select(1, _shape[1] - 1);

  auto f0 = f_face.select(-1, 0);
  auto f1 = f_face.select(-1, 1);
  auto f3 = f_face.select(-1, 3);
  auto f2 = f_face.select(-1, 2);
  auto f5 = f_face.select(-1, 5);
  auto f6 = f_face.select(-1, 6);

  auto velocity = torch::add(f2, f5).add_(f6).mul_(2.0).add_(f0).add_(f1).add_(f3);
  velocity.div_(_value).sub_(1.0);

  int bot_0 = _stencil._bottom[0].item<int>();
  int opp_0 = _stencil._op[bot_0].item<int>();

  u_face.select(-1, opp_0).copy_(f_face.select(-1, bot_0)).add_(velocity, -2.0 / 3.0 * _value);

  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    int bot_i = _stencil._bottom[i].item<int>();
    int opp_i = _stencil._op[bot_i].item<int>();
    double ex_opp = _stencil._ex[opp_i].item<double>();

    auto update =
        f_face.select(-1, bot_i) + 0.5 * ex_opp * (f3 - f1) - (1.0 / 6.0) * _value * velocity;
    u_face.select(-1, opp_i).copy_(update);
  }
}

void
LBMFixedZerothOrderBC::topBoundary()
{
  if (_stencil._q == 9)
    topBoundaryD2Q9();
  else
    mooseError("Top boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain");
}

void
LBMFixedZerothOrderBC::computeBuffer()
{
  _f_owned = _f;
  for (unsigned int d = 0; d < _dim; d++)
    _f_owned = _f_owned.narrow(d, _radius, _shape[d]);

  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
