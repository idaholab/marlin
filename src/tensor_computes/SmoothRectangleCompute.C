/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SmoothRectangleCompute.h"
// #include "MarlinUtils.h"
#include "TensorProblem.h"
#include <torch/torch.h>

registerMooseObject("MarlinApp", SmoothRectangleCompute);

InputParameters
SmoothRectangleCompute::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription(
      "Interpolate a value between the inside and outside of a rectangle smoothly.");
  params.addRequiredParam<Real>("x1", "The x coordinate of the lower left-hand corner of the box.");
  params.addRequiredParam<Real>("x2",
                                "The x coordinate of the upper right-hand corner of the box.");
  params.addRequiredParam<Real>("y1", "The x coordinate of the lower left-hand corner of the box.");
  params.addRequiredParam<Real>("y2",
                                "The x coordinate of the upper right-hand corner of the box.");
  params.addParam<Real>("z1", 0, "The z coordinate of the lower left-hand corner of the box.");
  params.addParam<Real>("z2", 0, "The z coordinate of the upper right-hand corner of the box.");
  MooseEnum interpolationFunction("COS TANH");
  params.addParam<MooseEnum>(
      "profile", interpolationFunction, "Functional dependence for the interface profile");
  params.addParam<Real>(
      "int_width", 0, "The width of the diffuse interface. Set to 0 for sharp interface.");
  params.addParam<Real>("inside", 1, "The value inside the rectangle.");
  params.addParam<Real>("outside", 0, "The value outside the rectangle.");
  return params;
}

SmoothRectangleCompute::SmoothRectangleCompute(const InputParameters & parameters)
  : TensorOperator(parameters),
    _x1(getParam<Real>("x1")),
    _x2(getParam<Real>("x2")),
    _y1(getParam<Real>("y1")),
    _y2(getParam<Real>("y2")),
    _z1(getParam<Real>("z1")),
    _z2(getParam<Real>("z2")),
    _interp_func(getParam<MooseEnum>("profile").getEnum<interpolationFunction>()),
    _int_width(getParam<Real>("int_width")),
    _inside(getParam<Real>("inside")),
    _outside(getParam<Real>("outside"))
{
  if (_int_width < 0.0)
    mooseError("Interface width must be a non-negative real number.");
}

void
SmoothRectangleCompute::computeBuffer()
{
  auto dim = _domain.getDim();
  auto h_box = torch::zeros(_tensor_problem.getShape(), MooseTensor::floatTensorOptions());

  // sharp interpolation
  if (_int_width <= 0.0)
  {
    auto h_x = (_x >= _x1) & (_x <= _x2);
    auto h_y = (dim >= 2) ? ((_y >= _y1) & (_y <= _y2)) : torch::ones_like(_y, torch::kBool);
    auto h_z = (dim == 3) ? ((_z >= _z1) & (_z <= _z2)) : torch::ones_like(_z, torch::kBool);
    // Reshape to the same dimensions for logical_and operation
    h_x = h_x.reshape({-1, 1, 1});
    h_y = h_y.reshape({1, -1, 1});
    h_z = h_z.reshape({1, 1, -1});

    // Combine the conditions
    auto combined_conditions = torch::logical_and(h_x, torch::logical_and(h_y, h_z)).squeeze();
    // Apply the combined conditions to h_box
    h_box.index_put_({combined_conditions}, 1.0);
  }
  else
  {
    // generate distances based on the right problem dimension

    switch (_interp_func)
    {
      case interpolationFunction::COS:
      {
        auto min_x = torch::minimum(_x - _x1, _x2 - _x).clamp(-_int_width / 2.0, _int_width / 2.0);
        auto min_y =
            (dim >= 2)
                ? torch::minimum(_y - _y1, _y2 - _y).clamp(-_int_width / 2.0, _int_width / 2.0)
                : torch::full_like(_y, _int_width / 2.0);
        auto min_z =
            (dim == 3)
                ? torch::minimum(_z - _z1, _z2 - _z).clamp(-_int_width / 2.0, _int_width / 2.0)
                : torch::full_like(_z, _int_width / 2.0);
        auto h_x = 0.5 + 0.5 * torch::sin(pi * min_x / _int_width);
        auto h_y = 0.5 + 0.5 * torch::sin(pi * min_y / _int_width);
        auto h_z = 0.5 + 0.5 * torch::sin(pi * min_z / _int_width);
        h_box = h_x.reshape({-1, 1, 1}) * h_y.reshape({1, -1, 1}) * h_z.reshape({1, 1, -1});
        break;
      }
      case interpolationFunction::TANH:
      {
        auto min_x = torch::minimum(_x - _x1, _x2 - _x);
        auto min_y =
            (dim >= 2) ? torch::minimum(_y - _y1, _y2 - _y) : torch::full_like(_y, 10 * _int_width);
        auto min_z = (dim == 3) ? torch::minimum(_z - _z1, _z2 - _z)
                                : torch::full_like(_z, 10 * _int_width / 2.0);
        auto h_x = 0.5 + 0.5 * torch::tanh(4 * min_x / _int_width);
        auto h_y = 0.5 + 0.5 * torch::tanh(4 * min_y / _int_width);
        auto h_z = 0.5 + 0.5 * torch::tanh(4 * min_z / _int_width);
        h_box = h_x.reshape({-1, 1, 1}) * h_y.reshape({1, -1, 1}) * h_z.reshape({1, 1, -1});
        break;
      }
    }
  }
  _u = (h_box * _inside + (1 - h_box) * _outside).squeeze();
}
