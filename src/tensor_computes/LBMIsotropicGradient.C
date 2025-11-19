/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMIsotropicGradient.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMIsotropicGradient);

InputParameters
LBMIsotropicGradient::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute isotropic gradient object.");
  params.addRequiredParam<TensorInputBufferName>("scalar_field",
                                                 "Scalar field to compute the gradient of");

  return params;
}

LBMIsotropicGradient::LBMIsotropicGradient(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), _scalar_field(getInputBuffer("scalar_field"))
{
  const unsigned int & dim = _domain.getDim();

  // Note: if D3Q19 stencil is used, isotropic gradient is NOT going to work,
  // because D3Q19 is NOT isotropic.

  if (_stencil._q == 19)
    mooseError("Isotropic gradient cannot be computed for D3Q19 stencil");

  _kernel = torch::zeros({3, 3, _domain.getDim()}, MooseTensor::floatTensorOptions());

  switch (dim)
  {
    case 3:
      mooseError("LBMIsotropicGradient is not implemented for 3D");
      break;
    case 2:
    {
      auto kernel_of_kernel =
          torch::index_select(_stencil._weights, 0, _stencil._reorder_indices).reshape({3, 3});
      auto ex3x3 = torch::index_select(_stencil._ex, 0, _stencil._reorder_indices).reshape({3, 3});
      auto ey3x3 = torch::index_select(_stencil._ey, 0, _stencil._reorder_indices).reshape({3, 3});

      _kernel.index_put_({Slice(), Slice(), 0}, kernel_of_kernel * ex3x3);
      _kernel.index_put_({Slice(), Slice(), 1}, kernel_of_kernel * ey3x3);

      _conv_options.bias(torch::Tensor()).stride({1, 1}).padding(0);
      break;
    }
  }
}

torch::Tensor
LBMIsotropicGradient::padScalarField()
{
  // because torch sucks at padding
  torch::Tensor right_pad_slice =
      _scalar_field.slice(1, _scalar_field.size(1) - _padding, _scalar_field.size(1));
  torch::Tensor left_pad_slice = _scalar_field.slice(1, 0, _padding);

  torch::Tensor padded_width = torch::cat({left_pad_slice, _scalar_field, right_pad_slice}, 1);

  torch::Tensor bottom_pad_slice =
      padded_width.slice(0, padded_width.size(0) - _padding, padded_width.size(0));
  torch::Tensor top_pad_slice = padded_width.slice(0, 0, _padding);

  torch::Tensor fully_padded_tensor =
      torch::cat({top_pad_slice, padded_width, bottom_pad_slice}, 0);

  return fully_padded_tensor;
}

void
LBMIsotropicGradient::computeBuffer()
{
  // check output buffer shape
  if ((unsigned int)_u.size(-1) != _domain.getDim())
    mooseError("Output buffer must have the same number of dimensions as the domain.");

  const unsigned int & dim = _domain.getDim();
  torch::Tensor kernel = _kernel.permute({2, 0, 1});
  kernel = kernel.unsqueeze(1);

  switch (dim)
  {
    case 3:
      mooseError("LBMIsotropicGradient is not implemented for 3D");
      break;
    case 2:
    {
      if (_scalar_field.dim() > 2)
        _scalar_field.squeeze_(-1);

      torch::Tensor input_field = padScalarField();

      input_field = input_field.unsqueeze(0).unsqueeze(0);

      torch::Tensor isotropic_gradient =
          torch::nn::functional::conv2d(input_field, kernel, _conv_options);

      isotropic_gradient = isotropic_gradient.squeeze(0);

      _u.index_put_({Slice(), Slice(), Slice(), 0},
                    isotropic_gradient.index({0, Slice(), Slice()}).unsqueeze(-1));
      _u.index_put_({Slice(), Slice(), Slice(), 1},
                    isotropic_gradient.index({1, Slice(), Slice()}).unsqueeze(-1));
      _u = _u / _lb_problem._cs2;
      break;
    }
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
