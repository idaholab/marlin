/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AnisotropicGBEnergy.h"

#include "MarlinUtils.h"

#include <cmath>
#include <limits>

registerMooseObject("MarlinApp", AnisotropicGBEnergy);

InputParameters
AnisotropicGBEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Evaluates anisotropic grain-boundary energy from a TorchScript model.");
  params.addRequiredParam<TensorInputBufferName>(
      "gb_gradient_buffer", "Input tensor buffer containing grain-boundary gradient vectors.");
  params.addRequiredParam<TensorOutputBufferName>(
      "dsigma_dn",
      "Output tensor buffer for the derivative of GB energy with respect to the normal tensor.");
  params.addRequiredParam<DataFileName>(
      "libtorch_model_file", "Path to the TorchScript file containing the GB energy model.");
  params.addRequiredRangeCheckedParam<Real>(
      "interface_width", "interface_width > 0", "Coarsest interface width in model.");
  return params;
}

AnisotropicGBEnergy::AnisotropicGBEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _gb_gradient_buffer(getInputBuffer("gb_gradient_buffer")),
    _dsigma_dn(getOutputBufferByName(getParam<TensorOutputBufferName>("dsigma_dn"))),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path))),
    _interface_width(getParam<Real>("interface_width"))
{
  const auto ref = torch::empty({0}, MooseTensor::floatTensorOptions());
  _surrogate->to(ref.device(), ref.scalar_type(), /* non_blocking = */ false);
  _surrogate->eval();

  // The gb_gradient_buffer magnitude falls below this threshold outside the
  // diffuse interface, where |grad eta| ~ (1/interface_width) / cosh(2*d/W)^2
  // for a tanh profile of half-width W/4 and d the signed distance from the
  // interface centre; d = W evaluates cosh(4), giving a negligible-gradient
  // cutoff one interface width out.
  _gradient_threshold = 2 / (cosh(4) * cosh(4)) / _interface_width;
}

void
AnisotropicGBEnergy::computeBuffer()
{
  if (_gb_gradient_buffer.dim() < 1)
    mooseError("gb_gradient_buffer must have at least one dimension.");

  if (_gb_gradient_buffer.size(-1) != 3)
    mooseError("gb_gradient_buffer must have trailing component dimension of size 3.");

  const auto batch_size = _gb_gradient_buffer.numel() / 3;
  std::vector<int64_t> normal_shape(_gb_gradient_buffer.sizes().begin(),
                                    _gb_gradient_buffer.sizes().end());
  std::vector<int64_t> output_shape(_gb_gradient_buffer.sizes().begin(),
                                    _gb_gradient_buffer.sizes().end() - 1);

  auto gb_gradient = _gb_gradient_buffer.reshape({batch_size, 3}).contiguous().detach();

  // Compute the valid (interface) mask on the full grid, then pre-filter to
  // interface points only before running the surrogate model.
  auto grad_mag =
      torch::sqrt((gb_gradient * gb_gradient).sum(/*dim=*/1, /*keepdim=*/true)); // [B, 1]
  auto valid_mask = (grad_mag.squeeze(1) >= _gradient_threshold);                // [B]

  auto valid_idx = torch::where(valid_mask)[0]; // [N_interface]
  const auto N_interface = valid_idx.size(0);

  // Outside the interface there is no gradient to differentiate, so
  // dsigma_dn stays zero there; the energy itself is set to the maximum
  // measured value below rather than left at zero, to avoid an artificial
  // low-energy region in the bulk.
  auto gamma_full = torch::full(
      {batch_size}, std::numeric_limits<float>::quiet_NaN(), gb_gradient.options());
  auto dsigma_dg_full = torch::zeros({batch_size, 3}, gb_gradient.options());

  if (N_interface > 0)
  {
    auto gb_grad_valid =
        gb_gradient.index_select(0, valid_idx).requires_grad_(true); // [N_interface, 3]

    // Recompute from the leaf so autograd differentiates through normalization
    auto grad_mag_valid = torch::sqrt(
        (gb_grad_valid * gb_grad_valid).sum(/*dim=*/1, /*keepdim=*/true)); // [N_interface, 1]

    auto n_hat = gb_grad_valid / grad_mag_valid; // [N_interface, 3]

    auto gamma_valid = _surrogate->forward({n_hat}).toTensor().reshape({N_interface});

    auto grads = torch::autograd::grad({gamma_valid.sum()},
                                       {gb_grad_valid},
                                       /*grad_outputs=*/{},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/false);
    auto dsigma_dg_valid = grads[0];

    gamma_valid = gamma_valid.detach();
    gamma_full.index_put_({valid_idx}, gamma_valid);
    dsigma_dg_full.index_put_({valid_idx}, dsigma_dg_valid.detach());

    // Fill the remaining non-interface points with the largest GBE actually
    // measured, to avoid an artificial low-energy region in the bulk.
    gamma_full = torch::where(torch::isnan(gamma_full), gamma_valid.max(), gamma_full);
  }
  else
    gamma_full.fill_(1.0);

  _u = gamma_full.reshape(output_shape);
  _dsigma_dn = dsigma_dg_full.reshape(normal_shape);
}
