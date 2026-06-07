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

  auto gb_gradient = _gb_gradient_buffer.reshape({batch_size, 3}).contiguous().detach().requires_grad_(true);

  // ── 1. Compute regularized magnitude and normalized directions ──────────
  // auto directions = gb_gradient;
  auto grad_mag_sq = (gb_gradient * gb_gradient).sum(/*dim=*/1, /*keepdim=*/true); // [B, 1]
  auto grad_mag = torch::sqrt(grad_mag_sq);                                      // [B, 1]

  auto valid_mask = (grad_mag >= _gradient_threshold).squeeze(1); // [B]
  auto mask3 = valid_mask.unsqueeze(1).expand({batch_size, 3}); // [B, 3]
  // ── 2. Detach n_hat — this is the autograd leaf, NOT directions ─────────
  //    Autograd will give us dσ/dn̂ cleanly, with no 1/|∇η| blowup
  auto n_hat = torch::where(mask3, gb_gradient / grad_mag, torch::zeros_like(gb_gradient)); // [B, 3]

  // ── 3. Forward pass through torchscript hull model ──────────────────────
  auto gamma = _surrogate->forward({n_hat}).toTensor().reshape({batch_size});


  // // ── 4. Get dσ/dn̂ via autograd — clean, no singularity ──────────────────
  auto grads = torch::autograd::grad({gamma.sum()},
                                     {gb_gradient},
                                     /*grad_outputs=*/{},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false,
                                     /*allow_unused=*/false);
  auto dσ_d_grad_eta = grads[0]; // [B, 3]

  // // ── 6. Apply valid mask ──────────────────────────────────────────────────
  auto masked_gamma = torch::where(valid_mask, gamma, torch::zeros_like(gamma));
  auto masked_dσ_dg = torch::where(mask3, dσ_d_grad_eta, torch::zeros_like(dσ_d_grad_eta));

  _u = masked_gamma.reshape(output_shape);
  _dsigma_dn = masked_dσ_dg.reshape(normal_shape);
}