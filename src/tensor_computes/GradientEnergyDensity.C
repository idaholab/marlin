/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GradientEnergyDensity.h"

registerMooseObject("MarlinApp", GradientEnergyDensity);

InputParameters
GradientEnergyDensity::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Computes the gradient energy density 0.5 * kappa * sum_i |grad eta_i|^2 from a set of "
      "gradient vector buffers.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "gradient_buffers",
      "Names of the gradient vector buffers (one per order parameter) to sum over.");
  params.addRequiredParam<Real>("kappa", "Gradient energy coefficient.");
  return params;
}

GradientEnergyDensity::GradientEnergyDensity(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _gradient_buffer_names(getParam<std::vector<TensorInputBufferName>>("gradient_buffers")),
    _kappa(getParam<Real>("kappa"))
{
  if (_gradient_buffer_names.empty())
    paramError("gradient_buffers", "At least one gradient buffer must be supplied.");
}

void
GradientEnergyDensity::computeBuffer()
{
  torch::Tensor sum_grad_sq;
  std::vector<int64_t> spatial_shape;

  for (const auto & name : _gradient_buffer_names)
  {
    const auto & gradient_buffer = getInputBufferByName(name);

    if (gradient_buffer.dim() < 1 || gradient_buffer.size(-1) != 3)
      mooseError(
          "Gradient buffer '", name, "' must have a trailing component dimension of size 3.");

    std::vector<int64_t> this_shape(gradient_buffer.sizes().begin(),
                                    gradient_buffer.sizes().end() - 1);

    if (!sum_grad_sq.defined())
    {
      spatial_shape = this_shape;
      sum_grad_sq = (gradient_buffer * gradient_buffer).sum(-1);
    }
    else
    {
      if (this_shape != spatial_shape)
        mooseError("Gradient buffer '", name, "' does not match the spatial shape of the others.");
      sum_grad_sq += (gradient_buffer * gradient_buffer).sum(-1);
    }
  }

  _u = 0.5 * _kappa * sum_grad_sq;
}
