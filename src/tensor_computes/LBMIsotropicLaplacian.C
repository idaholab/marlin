/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMIsotropicLaplacian.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMIsotropicLaplacian);

InputParameters
LBMIsotropicLaplacian::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute isotropic Laplacian object.");
  params.addRequiredParam<TensorInputBufferName>("scalar_field",
                                                 "Scalar field to compute the Laplacian of");

  return params;
}

LBMIsotropicLaplacian::LBMIsotropicLaplacian(const InputParameters & parameters)
  : LBMIsotropicGradient(parameters)
{
  const unsigned int & dim = _domain.getDim();

  // Note: if D3Q19 stencil is used, isotropic gradient is NOT going to work,
  // because D3Q19 is NOT isotropic.

  if (_stencil._q == 19)
    mooseError("Isotropic Laplacian cannot be computed for D3Q19 stencil");

  _kernel = torch::zeros({3, 3}, MooseTensor::floatTensorOptions());

  switch (dim)
  {
    case 3:
      mooseError("LBMIsotropicLaplacian is not implemented for 3D");
      break;
    case 2:
    {
      _kernel =
          torch::index_select(_stencil._weights, 0, _stencil._reorder_indices).reshape({3, 3});
      _conv_options.bias(torch::Tensor()).stride({1, 1}).padding(0);
      break;
    }
  }
}

void
LBMIsotropicLaplacian::computeBuffer()
{
  const unsigned int & dim = _domain.getDim();
  torch::Tensor kernel = _kernel.view({1, 1, 3, 3});

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

      torch::Tensor isotropic_Laplacian_1 =
          torch::nn::functional::conv2d(input_field, kernel, _conv_options);

      isotropic_Laplacian_1 = 2.0 * isotropic_Laplacian_1.squeeze(0).squeeze(0);

      auto isotropic_Laplacian_2 =
          2.0 *
          torch::sum(_scalar_field.unsqueeze(-1) * _stencil._weights.unsqueeze(0).unsqueeze(0), -1);

      _u = (isotropic_Laplacian_1.unsqueeze(-1) - isotropic_Laplacian_2.unsqueeze(-1)) /
           _lb_problem._cs2;
      break;
    }
  }
}
