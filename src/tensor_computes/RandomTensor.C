/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RandomTensor.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"
#include <core/DeviceType.h>

registerMooseObject("MarlinApp", RandomTensor);

InputParameters
RandomTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Uniform random IC with values between `min` and `max`.");
  params.addRequiredParam<Real>("min", "Minimum value.");
  params.addRequiredParam<Real>("max", "Maximum value.");
  params.addParam<int>("seed", "Random number seed.");
  params.addParam<bool>("generate_on_cpu",
                        true,
                        "To ensure reproducibility across devices it is recommended to generate "
                        "random tensors on the CPU.");
  return params;
}

RandomTensor::RandomTensor(const InputParameters & parameters)
  : TensorOperator<>(parameters), _generate_on_cpu(getParam<bool>("generate_on_cpu"))
{
}

void
RandomTensor::computeBuffer()
{
  const auto min = getParam<Real>("min");
  const auto max = getParam<Real>("max");
  if (isParamValid("seed"))
    torch::manual_seed(getParam<int>("seed"));

  if (_generate_on_cpu)
  {
    _u = torch::rand(_tensor_problem.getShape(),
                     MooseTensor::floatTensorOptions().device(torch::kCPU))
                 .to(MooseTensor::floatTensorOptions()) *
             (max - min) +
         min;
  }
  else
    _u = torch::rand(_tensor_problem.getShape(), MooseTensor::floatTensorOptions()) * (max - min) +
         min;
}
