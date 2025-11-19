/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseFunctionTensor.h"
#include "Function.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", MooseFunctionTensor);

InputParameters
MooseFunctionTensor::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Map a MooseFunction to a tensor.");
  params.addRequiredParam<FunctionName>("function", "Function to map.");
  // params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  return params;
}

MooseFunctionTensor::MooseFunctionTensor(const InputParameters & parameters)
  : TensorOperator(parameters), FunctionInterface(this), _func(getFunction("function"))
{
}

void
MooseFunctionTensor::computeBuffer()
{
  auto buffer = torch::zeros(_tensor_problem.getShape(), torch::kDouble);

  const auto & n = _domain.getGridSize();
  const auto & dx = _domain.getGridSpacing();

  switch (_domain.getDim())
  {
    {
      case 1:
        auto b = buffer.accessor<double, 1>();
        for (const auto i : make_range(n[0]))
          b[i] = _func.value(0, Point(i * dx(0) + dx(0) / 2.0, 0.0, 0.0));
        break;
    }
    case 2:
    {
      auto b = buffer.accessor<double, 2>();
      for (const auto j : make_range(n[1]))
        for (const auto i : make_range(n[0]))
          b[i][j] = _func.value(0, Point(i * dx(0) + dx(0) / 2.0, j * dx(1) + dx(1) / 2.0, 0.0));
      break;
    }
    case 3:
    {
      auto b = buffer.accessor<double, 3>();
      for (const auto k : make_range(n[2]))
        for (const auto j : make_range(n[1]))
          for (const auto i : make_range(n[0]))
            b[i][j][k] = _func.value(
                0,
                Point(i * dx(0) + dx(0) / 2.0, j * dx(1) + dx(1) / 2.0, k * dx(2) + dx(2) / 2.0));
      break;
    }
    default:
      mooseError("Unsupported dimension");
  }

  _u = buffer.to(MooseTensor::floatTensorOptions());
}
