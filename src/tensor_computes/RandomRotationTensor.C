/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RandomRotationTensor.h"
#include "MarlinUtils.h"

#include <random>

registerMooseObject("MarlinApp", RandomRotationTensor);

InputParameters
RandomRotationTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Uniform random 3x3 rotation matrix (constant over the domain).");
  params.addParam<int>("seed", "Random number seed.");
  params.addParam<bool>("generate_on_cpu",
                        true,
                        "To ensure reproducibility across devices it is recommended to generate "
                        "random tensors on the CPU.");
  return params;
}

RandomRotationTensor::RandomRotationTensor(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _generate_on_cpu(getParam<bool>("generate_on_cpu")),
    _has_seed(isParamValid("seed")),
    _seed(_has_seed ? getParam<int>("seed") : 0)
{
}

void
RandomRotationTensor::computeBuffer()
{
  if (_u.dim() < static_cast<int64_t>(_dim) + 2)
    mooseError("RandomRotationTensor requires a buffer with value_dimensions = '3 3'.");

  if (_u.size(_dim) != 3 || _u.size(_dim + 1) != 3)
    mooseError("RandomRotationTensor requires a buffer with value_dimensions = '3 3'.");

  std::mt19937 gen;
  if (_has_seed)
    gen.seed(static_cast<unsigned int>(_seed));
  else
    gen.seed(std::random_device{}());

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const double u1 = dist(gen);
  const double u2 = dist(gen);
  const double u3 = dist(gen);

  const double sqrt1_u1 = std::sqrt(1.0 - u1);
  const double sqrt_u1 = std::sqrt(u1);
  const double theta1 = 2.0 * libMesh::pi * u2;
  const double theta2 = 2.0 * libMesh::pi * u3;

  const double x = sqrt1_u1 * std::sin(theta1);
  const double y = sqrt1_u1 * std::cos(theta1);
  const double z = sqrt_u1 * std::sin(theta2);
  const double w = sqrt_u1 * std::cos(theta2);

  const double r11 = 1.0 - 2.0 * (y * y + z * z);
  const double r12 = 2.0 * (x * y - z * w);
  const double r13 = 2.0 * (x * z + y * w);
  const double r21 = 2.0 * (x * y + z * w);
  const double r22 = 1.0 - 2.0 * (x * x + z * z);
  const double r23 = 2.0 * (y * z - x * w);
  const double r31 = 2.0 * (x * z - y * w);
  const double r32 = 2.0 * (y * z + x * w);
  const double r33 = 1.0 - 2.0 * (x * x + y * y);

  auto opts = MooseTensor::floatTensorOptions();
  if (_generate_on_cpu)
    opts = opts.device(torch::kCPU);

  auto R = torch::tensor({{r11, r12, r13}, {r21, r22, r23}, {r31, r32, r33}}, opts);
  R = R.to(_u.options());

  auto expanded = R;
  for (unsigned int d = 0; d < _dim; ++d)
    expanded = expanded.unsqueeze(0);
  _u = expanded.expand(_u.sizes());
}
