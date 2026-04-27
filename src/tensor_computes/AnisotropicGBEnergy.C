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

registerMooseObject("MarlinApp", AnisotropicGBEnergy);

InputParameters
AnisotropicGBEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Evaluates anisotropic grain-boundary energy from a TorchScript model.");
  params.addRequiredParam<TensorInputBufferName>(
      "gb_normal_buffer", "Input tensor buffer containing grain-boundary normal vectors.");
  params.addRequiredParam<TensorOutputBufferName>(
      "dsigma_dn", "Output tensor buffer for the derivative of GB energy with respect to the normal tensor.");
  params.addRequiredParam<DataFileName>(
      "libtorch_model_file", "Path to the TorchScript file containing the GB energy model.");
  return params;
}

AnisotropicGBEnergy::AnisotropicGBEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _gb_normal_buffer(getInputBuffer("gb_normal_buffer")),
    _dsigma_dn(getOutputBufferByName(getParam<TensorOutputBufferName>("dsigma_dn"))),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path)))
{
  const auto ref = torch::empty({0}, MooseTensor::floatTensorOptions());
  _surrogate->to(ref.device(), ref.scalar_type(), /* non_blocking = */ false);
  _surrogate->eval();
}

double
AnisotropicGBEnergy::epsilonFor(const torch::ScalarType dtype) const
{
  switch (dtype)
  {
    case torch::kFloat32:
      return std::numeric_limits<float>::epsilon();
    case torch::kFloat64:
      return std::numeric_limits<double>::epsilon();
    default:
      mooseError("AnisotropicGBEnergy only supports float32 and float64 input tensors.");
  }
}

void
AnisotropicGBEnergy::computeBuffer()
{
  if (_gb_normal_buffer.dim() < 1)
    mooseError("gb_normal_buffer must have at least one dimension.");

  if (_gb_normal_buffer.size(-1) != 3)
    mooseError("gb_normal_buffer must have trailing component dimension of size 3.");

  const auto batch_size = _gb_normal_buffer.numel() / 3;
  std::vector<int64_t> normal_shape(
      _gb_normal_buffer.sizes().begin(), _gb_normal_buffer.sizes().end());
  std::vector<int64_t> output_shape(
      _gb_normal_buffer.sizes().begin(), _gb_normal_buffer.sizes().end() - 1);

  // ── 1. Compute regularized magnitude and normalized directions ──────────
  auto directions     = _gb_normal_buffer.reshape({batch_size, 3}).contiguous();
  auto grad_mag_sq    = (directions * directions).sum(/*dim=*/1, /*keepdim=*/true); // [B, 1]
  auto grad_mag       = torch::sqrt(grad_mag_sq);                                   // [B, 1]
  auto grad_mag_reg   = torch::sqrt(grad_mag_sq + 1e-6);                            // [B, 1] — regularized

  const auto threshold = 1e-3;
  auto valid_mask      = (grad_mag > threshold).squeeze(1);                         // [B]

  // ── 2. Detach n_hat — this is the autograd leaf, NOT directions ─────────
  //    Autograd will give us dσ/dn̂ cleanly, with no 1/|∇η| blowup
  auto n_hat = (directions / grad_mag_reg).detach().requires_grad_(true);           // [B, 3]

  // ── 3. Forward pass through torchscript hull model ──────────────────────
  auto gamma = _surrogate->forward({n_hat}).toTensor().reshape({batch_size});

  // ── 4. Get dσ/dn̂ via autograd — clean, no singularity ──────────────────
  auto grads = torch::autograd::grad(
      {gamma.sum()}, {n_hat},
      /*grad_outputs=*/{},
      /*retain_graph=*/false,
      /*create_graph=*/false,
      /*allow_unused=*/false);
  auto dσ_dn = grads[0];                                                            // [B, 3]

  // ── 5. Apply projection Jacobian: (I - n̂⊗n̂) / |∇η|_reg ────────────────
  //    dσ/d(∇η) = (I - n̂⊗n̂) / |∇η|_reg  ·  dσ/dn̂
  //
  //    n̂⊗n̂ term: (n̂·dσ/dn̂) * n̂
  auto n_hat_detached  = n_hat.detach();                                            // [B, 3]
  auto ndot            = (n_hat_detached * dσ_dn).sum(/*dim=*/1, /*keepdim=*/true);// [B, 1]
  auto dσ_dg           = (dσ_dn - ndot * n_hat_detached) / grad_mag_reg;           // [B, 3]

  // ── 6. Apply valid mask ──────────────────────────────────────────────────
  auto mask3           = valid_mask.unsqueeze(1).expand({batch_size, 3});           // [B, 3]
  auto masked_gamma    = torch::where(valid_mask,  gamma,  torch::zeros_like(gamma));
  auto masked_dσ_dg    = torch::where(mask3,       dσ_dg,  torch::zeros_like(dσ_dg));

  _u        = masked_gamma.reshape(output_shape);
  _dsigma_dn = masked_dσ_dg.reshape(normal_shape);
}