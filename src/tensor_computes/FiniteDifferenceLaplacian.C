/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceLaplacian.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", FiniteDifferenceLaplacian);

InputParameters
FiniteDifferenceLaplacian::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The input buffer");
  params.addClassDescription("Computes the Laplacian using finite differences.");
  return params;
}

FiniteDifferenceLaplacian::FiniteDifferenceLaplacian(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters), _u_in(getInputBuffer("input", 1))
{
}

void
FiniteDifferenceLaplacian::computeBuffer()
{
  const auto & grid_spacing = _domain.getGridSpacing();

  // Use convolution for efficient stencil application
  // Construct kernel
  torch::Tensor kernel;
  auto options = _u_in.options();

  if (_dim == 1)
  {
    auto dx = grid_spacing(0);
    kernel = torch::tensor({1.0, -2.0, 1.0}, options) / (dx * dx);
    kernel = kernel.view({1, 1, 3});

    auto input = _u_in.view({1, 1, _u_in.size(0)});
    auto result = torch::nn::functional::conv1d(
        input, kernel, torch::nn::functional::Conv1dFuncOptions().padding(1));
    _u.copy_(result.view(_u.sizes()));
  }
  else if (_dim == 2)
  {
    auto dx = grid_spacing(0);
    auto dy = grid_spacing(1);

    kernel = torch::zeros({3, 3}, options);
    // Center
    kernel[1][1] = -2.0 / (dx * dx) - 2.0 / (dy * dy);
    // X neighbors (rows)
    kernel[0][1] = 1.0 / (dx * dx);
    kernel[2][1] = 1.0 / (dx * dx);
    // Y neighbors (cols)
    kernel[1][0] = 1.0 / (dy * dy);
    kernel[1][2] = 1.0 / (dy * dy);

    kernel = kernel.view({1, 1, 3, 3});

    auto input = _u_in.view({1, 1, _u_in.size(0), _u_in.size(1)});
    auto result = torch::nn::functional::conv2d(
        input, kernel, torch::nn::functional::Conv2dFuncOptions().padding(1));
    _u.copy_(result.view(_u.sizes()));
  }
  else if (_dim == 3)
  {
    auto dx = grid_spacing(0);
    auto dy = grid_spacing(1);
    auto dz = grid_spacing(2);

    kernel = torch::zeros({3, 3, 3}, options);
    // Center
    kernel[1][1][1] = -2.0 / (dx * dx) - 2.0 / (dy * dy) - 2.0 / (dz * dz);
    // X neighbors (dim 0)
    kernel[0][1][1] = 1.0 / (dx * dx);
    kernel[2][1][1] = 1.0 / (dx * dx);
    // Y neighbors (dim 1)
    kernel[1][0][1] = 1.0 / (dy * dy);
    kernel[1][2][1] = 1.0 / (dy * dy);
    // Z neighbors (dim 2)
    kernel[1][1][0] = 1.0 / (dz * dz);
    kernel[1][1][2] = 1.0 / (dz * dz);

    kernel = kernel.view({1, 1, 3, 3, 3});

    auto input = _u_in.view({1, 1, _u_in.size(0), _u_in.size(1), _u_in.size(2)});
    auto result = torch::nn::functional::conv3d(
        input, kernel, torch::nn::functional::Conv3dFuncOptions().padding(1));
    _u.copy_(result.view(_u.sizes()));
  }
}
