/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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

  auto reordered_weights = torch::index_select(_stencil._weights, 0, _stencil._reorder_indices);

  if (dim == 3)
    _kernel = reordered_weights.reshape({3, 3, 3});
  else
    _kernel = reordered_weights.reshape({3, 3});
}

void
LBMIsotropicLaplacian::computeBuffer()
{
  const unsigned int dim = _domain.getDim();
  _lb_problem.exchangeGhostLayers(getParam<TensorInputBufferName>("scalar_field"), _radius);

  torch::Tensor input_field = prepareInputField().unsqueeze(0).unsqueeze(0);

  // Convolve with weight kernel
  torch::Tensor L1;
  if (dim == 3)
  {
    L1 = stencilConvolve3D(input_field.squeeze(0).squeeze(0), _kernel.reshape({_stencil._q, 1LL}));
  }
  else
  {
    auto kernel = _kernel.view({1, 1, 3, 3});
    L1 = torch::nn::functional::conv2d(input_field, kernel, _conv2d_options);
  }
  L1 = 2.0 * L1.squeeze(0).squeeze(0);

  // Weighted sum at each point: phi(x) * sum(w_i)
  auto L2 =
      2.0 *
      torch::sum(_scalar_field.unsqueeze(-1) * _stencil._weights.unsqueeze(0).unsqueeze(0), -1);

  _u_owned = ownedView(_u);
  auto result = L1 - ownedView(L2);
  if (dim == 2)
    result = result.unsqueeze(-1);
  _u_owned.copy_(result);
  _u_owned.div_(_lb_problem._cs2);
}
