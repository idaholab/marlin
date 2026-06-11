/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AnisotropicGBEnergy.h"

#include "MarlinUtils.h"

#include <limits>

#include <cmath>

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
  params.addRequiredParam<Real>("interface_width", "Coarsest interface width in model.");
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

  // ── 1. Compute valid mask on full grid (cheap) ────────────────────────
  auto grad_mag =
      torch::sqrt((gb_gradient * gb_gradient).sum(/*dim=*/1, /*keepdim=*/true)); // [B, 1]
  auto valid_mask = (grad_mag.squeeze(1) >= _gradient_threshold);                // [B]

  // ── 2. Pre-filter to interface points only ────────────────────────────
  auto valid_idx = torch::where(valid_mask)[0]; // [N_interface]
  const auto N_interface = valid_idx.size(0);

  // Initialize full-grid outputs to zero
  auto gamma_full = torch::zeros({batch_size}, gb_gradient.options());
  auto dσ_dg_full = torch::zeros({batch_size, 3}, gb_gradient.options());

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
    auto dσ_dg_valid = grads[0];

    gamma_full.index_put_({valid_idx}, gamma_valid.detach());
    dσ_dg_full.index_put_({valid_idx}, dσ_dg_valid.detach());
  }

  _u = gamma_full.reshape(output_shape);
  _dsigma_dn = dσ_dg_full.reshape(normal_shape);
}