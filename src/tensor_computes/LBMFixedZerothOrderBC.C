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
  torch::Tensor velocity =
      1.0 - (_f.index({0, Slice(), Slice(), 0}) + _f.index({0, Slice(), Slice(), 2}) +
             _f.index({0, Slice(), Slice(), 4}) +
             2 * (_f.index({0, Slice(), Slice(), 3}) + _f.index({0, Slice(), Slice(), 6}) +
                  _f.index({0, Slice(), Slice(), 7}))) /
                _value;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                _f.index({0, Slice(), Slice(), opposite_dir}) + 2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_(
        {0, Slice(), Slice(), _stencil._left[i]},
        _f.index({0, Slice(), Slice(), opposite_dir}) -
            0.5 * _stencil._ey[_stencil._left[i]] *
                (_f.index({0, Slice(), Slice(), 2}) - _f.index({0, Slice(), Slice(), 4})) +
            1.0 / 6.0 * _value * velocity);
  }
}

void
LBMFixedZerothOrderBC::leftBoundary()
{
  if (_stencil._q == 9)
    leftBoundaryD2Q9();
  else
  {
    torch::Tensor velocity =
        1.0 - (torch::sum(_f.index({0, Slice(), Slice(), -_stencil._neutral_x}), -1) +
               2 * torch::sum(_f.index({0, Slice(), Slice(), _stencil._right}), -1)) /
                  _value;

    _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                  _f.index({0, Slice(), Slice(), _stencil._right[0]}) +
                      2.0 * _stencil._weights[_stencil._left[0]] / _lb_problem._cs2 * _value *
                          velocity);

    for (unsigned int i = 1; i < _stencil._left.size(0); i++)
    {
      _u.index_put_({0, Slice(), Slice(), _stencil._left[i]},
                    _f.index({0, Slice(), Slice(), _stencil._right[i]}) +
                        2.0 * _stencil._weights[_stencil._left[i]] / _lb_problem._cs2 * _value *
                            velocity);
    }
  }
}

void
LBMFixedZerothOrderBC::rightBoundaryD2Q9()
{
  torch::Tensor velocity = (_f.index({_shape[0] - 1, Slice(), Slice(), 0}) +
                            _f.index({_shape[0] - 1, Slice(), Slice(), 2}) +
                            _f.index({_shape[0] - 1, Slice(), Slice(), 4}) +
                            2 * (_f.index({_shape[0] - 1, Slice(), Slice(), 1}) +
                                 _f.index({_shape[0] - 1, Slice(), Slice(), 5}) +
                                 _f.index({_shape[0] - 1, Slice(), Slice(), 8}))) /
                               _value -
                           1.0;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({_shape[0] - 1, Slice(), Slice(), opposite_dir},
                _f.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                    2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_shape[0] - 1, Slice(), Slice(), opposite_dir},
                  _f.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[i]}) +
                      0.5 * _stencil._ey[opposite_dir] *
                          (_f.index({_shape[0] - 1, Slice(), Slice(), 4}) -
                           _f.index({_shape[0] - 1, Slice(), Slice(), 2})) -
                      1.0 / 6.0 * _value * velocity);
  }
}

void
LBMFixedZerothOrderBC::rightBoundary()
{
  if (_stencil._q == 9)
    rightBoundaryD2Q9();
  else
  {
    torch::Tensor velocity =
        (torch::sum(_f.index({_shape[0] - 1, Slice(), Slice(), -_stencil._neutral_x}), -1) +
         2 * torch::sum(_f.index({_shape[0] - 1, Slice(), Slice(), _stencil._left}), -1)) /
            _value -
        1.0;

    _u.index_put_({_shape[0] - 1, Slice(), Slice(), _stencil._right[0]},
                  _f.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                      2.0 * _stencil._weights[_stencil._right[0]] / _lb_problem._cs2 * _value *
                          velocity);

    for (unsigned int i = 1; i < _stencil._right.size(0); i++)
    {
      _u.index_put_({_shape[0] - 1, Slice(), Slice(), _stencil._right[i]},
                    _f.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[i]}) -
                        2.0 * _stencil._weights[_stencil._right[i]] / _lb_problem._cs2 * _value *
                            velocity);
    }
  }
}

void
LBMFixedZerothOrderBC::bottomBoundaryD2Q9()
{
  torch::Tensor velocity =
      1.0 - (_f.index({Slice(), 0, Slice(), 0}) + _f.index({Slice(), 0, Slice(), 1}) +
             _f.index({Slice(), 0, Slice(), 3}) +
             2 * (_f.index({Slice(), 0, Slice(), 4}) + _f.index({Slice(), 0, Slice(), 7}) +
                  _f.index({Slice(), 0, Slice(), 8}))) /
                _value;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), 0, Slice(), _stencil._bottom[0]},
                _f.index({Slice(), 0, Slice(), opposite_dir}) + 2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_(
        {Slice(), 0, Slice(), _stencil._bottom[i]},
        _f.index({Slice(), 0, Slice(), opposite_dir}) -
            0.5 * _stencil._ex[_stencil._bottom[i]] *
                (_f.index({Slice(), 0, Slice(), 1}) - _f.index({Slice(), 0, Slice(), 3})) +
            1.0 / 6.0 * _value * velocity);
  }
}

void
LBMFixedZerothOrderBC::bottomBoundary()
{
  // TBD
  if (_stencil._q == 9)
    bottomBoundaryD2Q9();
  else
    mooseError("Bottom boundary is not implemented, but it can be replaced by any other boundary "
               "by rotating the domain");
}

void
LBMFixedZerothOrderBC::topBoundaryD2Q9()
{
  torch::Tensor velocity = (_f.index({Slice(), _shape[1] - 1, Slice(), 0}) +
                            _f.index({Slice(), _shape[1] - 1, Slice(), 1}) +
                            _f.index({Slice(), _shape[1] - 1, Slice(), 3}) +
                            2 * (_f.index({Slice(), _shape[1] - 1, Slice(), 2}) +
                                 _f.index({Slice(), _shape[1] - 1, Slice(), 5}) +
                                 _f.index({Slice(), _shape[1] - 1, Slice(), 6}))) /
                               _value -
                           1.0;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), _shape[1] - 1, Slice(), opposite_dir},
                _f.index({Slice(), _shape[1] - 1, Slice(), _stencil._bottom[0]}) -
                    2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({Slice(), _shape[1] - 1, Slice(), opposite_dir},
                  _f.index({Slice(), _shape[1] - 1, Slice(), _stencil._bottom[i]}) +
                      0.5 * _stencil._ex[opposite_dir] *
                          (_f.index({Slice(), _shape[1] - 1, Slice(), 3}) -
                           _f.index({Slice(), _shape[1] - 1, Slice(), 1})) -
                      1.0 / 6.0 * _value * velocity);
  }
}

void
LBMFixedZerothOrderBC::topBoundary()
{
  // TBD
  if (_stencil._q == 9)
    topBoundaryD2Q9();
  else
    mooseError("Top boundary is not implemented, but it can be replaced by any other boundary by "
               "rotating the domain");
}

void
LBMFixedZerothOrderBC::computeBuffer()
{
  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u, 0);
}
