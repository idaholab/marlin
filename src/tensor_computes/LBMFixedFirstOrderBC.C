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
  auto rank = _domain.comm().rank();
  std::array<int64_t, 3> begin, end;
  _domain.getLocalBounds(rank, begin, end);
  auto n_global = _domain.getGridSize();
  torch::Tensor u_x_perturbed =
      torch::zeros({end[1] - begin[1], 1}, MooseTensor::floatTensorOptions());

  if (_perturb)
  {
    Real deltaU = 1.0e-6 * _value;
    torch::Tensor y_coords =
        torch::arange(begin[1], end[1], MooseTensor::floatTensorOptions()).unsqueeze(1) /
        n_global[1];
    u_x_perturbed = _value + deltaU * torch::sin(y_coords * 2.0 * M_PI);
  }
  else
    u_x_perturbed.fill_(_value);

  torch::Tensor density =
      1.0 / (1.0 - u_x_perturbed) *
      (_f_owned.index({0, Slice(), Slice(), 0}) + _f_owned.index({0, Slice(), Slice(), 2}) +
       _f_owned.index({0, Slice(), Slice(), 4}) +
       2.0 * (_f_owned.index({0, Slice(), Slice(), 3}) + _f_owned.index({0, Slice(), Slice(), 6}) +
              _f_owned.index({0, Slice(), Slice(), 7})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u_owned.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                      _f_owned.index({0, Slice(), Slice(), opposite_dir}) +
                          2.0 / 3.0 * density * u_x_perturbed);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u_owned.index_put_({0, Slice(), Slice(), _stencil._left[i]},
                        _f_owned.index({0, Slice(), Slice(), opposite_dir}) -
                            0.5 * _stencil._ey[_stencil._left[i]] *
                                (_f_owned.index({0, Slice(), Slice(), 2}) -
                                 _f_owned.index({0, Slice(), Slice(), 4})) +
                            1.0 / 6.0 * density * u_x_perturbed);
  }
}

void
LBMFixedFirstOrderBC::leftBoundary()
{
  if (_stencil._q == 9)
    leftBoundaryD2Q9(); // higher order specialization for D2Q9
  else
  {
    torch::Tensor density =
        1.0 / (1.0 - _value) *
        (torch::sum(_f_owned.index({0, Slice(), Slice(), -_stencil._neutral_x}), -1) +
         2 * torch::sum(_f_owned.index({0, Slice(), Slice(), _stencil._right}), -1));

    _u_owned.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                        _f_owned.index({0, Slice(), Slice(), _stencil._right[0]}) +
                            2.0 * _stencil._weights[_stencil._left[0]] / _lb_problem._cs2 * _value *
                                density);

    for (unsigned int i = 1; i < _stencil._left.size(0); i++)
    {
      _u_owned.index_put_({0, Slice(), Slice(), _stencil._left[i]},
                          _f_owned.index({0, Slice(), Slice(), _stencil._right[i]}) +
                              2.0 * _stencil._weights[_stencil._left[i]] / _lb_problem._cs2 *
                                  _value * density);
    }
  }
}

void
LBMFixedFirstOrderBC::rightBoundaryD2Q9()
{
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (_f_owned.index({_shape[0] - 1, Slice(), Slice(), 0}) +
                           _f_owned.index({_shape[0] - 1, Slice(), Slice(), 2}) +
                           _f_owned.index({_shape[0] - 1, Slice(), Slice(), 4}) +
                           2 * (_f_owned.index({_shape[0] - 1, Slice(), Slice(), 1}) +
                                _f_owned.index({_shape[0] - 1, Slice(), Slice(), 5}) +
                                _f_owned.index({_shape[0] - 1, Slice(), Slice(), 8})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), opposite_dir},
                      _f_owned.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                          2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), opposite_dir},
                        _f_owned.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[i]}) +
                            0.5 * _stencil._ey[opposite_dir] *
                                (_f_owned.index({_shape[0] - 1, Slice(), Slice(), 4}) -
                                 _f_owned.index({_shape[0] - 1, Slice(), Slice(), 2})) -
                            1.0 / 6.0 * density * _value);
  }
}

void
LBMFixedFirstOrderBC::rightBoundary()
{
  if (_stencil._q == 9)
    rightBoundaryD2Q9(); // higher order specialization for D2Q9
  else
  {
    torch::Tensor density =
        1.0 / (1.0 + _value) *
        (torch::sum(_f_owned.index({_shape[0] - 1, Slice(), Slice(), -_stencil._neutral_x}), -1) +
         2 * torch::sum(_f_owned.index({_shape[0] - 1, Slice(), Slice(), _stencil._left}), -1));

    _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), _stencil._right[0]},
                        _f_owned.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                            2.0 * _stencil._weights[_stencil._right[0]] / _lb_problem._cs2 *
                                _value * density);

    for (unsigned int i = 1; i < _stencil._right.size(0); i++)
    {
      _u_owned.index_put_({_shape[0] - 1, Slice(), Slice(), _stencil._right[i]},
                          _f_owned.index({_shape[0] - 1, Slice(), Slice(), _stencil._left[i]}) -
                              2.0 * _stencil._weights[_stencil._right[i]] / _lb_problem._cs2 *
                                  _value * density);
    }
  }
}

void
LBMFixedFirstOrderBC::bottomBoundaryD2Q9()
{
  torch::Tensor density =
      1.0 / (1.0 - _value) *
      (_f_owned.index({Slice(), 0, Slice(), 0}) + _f_owned.index({Slice(), 0, Slice(), 1}) +
       _f_owned.index({Slice(), 0, Slice(), 3}) +
       2 * (_f_owned.index({Slice(), 0, Slice(), 4}) + _f_owned.index({Slice(), 0, Slice(), 7}) +
            _f_owned.index({Slice(), 0, Slice(), 8})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u_owned.index_put_({Slice(), 0, Slice(), _stencil._bottom[0]},
                      _f_owned.index({Slice(), 0, Slice(), opposite_dir}) +
                          2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u_owned.index_put_({Slice(), 0, Slice(), _stencil._bottom[i]},
                        _f_owned.index({Slice(), 0, Slice(), opposite_dir}) -
                            0.5 * _stencil._ex[_stencil._bottom[i]] *
                                (_f_owned.index({Slice(), 0, Slice(), 1}) -
                                 _f_owned.index({Slice(), 0, Slice(), 3})) +
                            1.0 / 6.0 * density * _value);
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
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (_f_owned.index({Slice(), _shape[1] - 1, Slice(), 0}) +
                           _f_owned.index({Slice(), _shape[1] - 1, Slice(), 1}) +
                           _f_owned.index({Slice(), _shape[1] - 1, Slice(), 3}) +
                           2 * (_f_owned.index({Slice(), _shape[1] - 1, Slice(), 2}) +
                                _f_owned.index({Slice(), _shape[1] - 1, Slice(), 5}) +
                                _f_owned.index({Slice(), _shape[1] - 1, Slice(), 6})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u_owned.index_put_({Slice(), _shape[1] - 1, Slice(), opposite_dir},
                      _f_owned.index({Slice(), _shape[1] - 1, Slice(), _stencil._bottom[0]}) -
                          2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u_owned.index_put_({Slice(), _shape[1] - 1, Slice(), opposite_dir},
                        _f_owned.index({Slice(), _shape[1] - 1, Slice(), _stencil._bottom[i]}) +
                            0.5 * _stencil._ex[opposite_dir] *
                                (_f_owned.index({Slice(), _shape[1] - 1, Slice(), 3}) -
                                 _f_owned.index({Slice(), _shape[1] - 1, Slice(), 1})) -
                            1.0 / 6.0 * density * _value);
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
