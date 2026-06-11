/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PairwiseAnisotropicGBEnergy.h"

#include "MarlinUtils.h"

#include <limits>

#include <cmath>

registerMooseObject("MarlinApp", PairwiseAnisotropicGBEnergy);

InputParameters
PairwiseAnisotropicGBEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Evaluates anisotropic grain-boundary energy from a TorchScript model.");
  params.addRequiredParam<TensorInputBufferName>(
      "grad_grain1_buffer", "Input tensor buffer containing gradient vectors for left grain.");
  params.addRequiredParam<TensorInputBufferName>(
      "grad_grain2_buffer", "Input tensor buffer containing gradient vectors for right grain.");
  params.addRequiredParam<TensorOutputBufferName>(
      "dsigma_dgrad_grain1",
      "Output tensor buffer for the derivative of GB energy with respect to the left grain "
      "gradient.");
  params.addRequiredParam<TensorOutputBufferName>(
      "dsigma_dgrad_grain2",
      "Output tensor buffer for the derivative of GB energy with respect to the right grain "
      "gradient.");
  params.addRequiredParam<DataFileName>(
      "libtorch_model_file", "Path to the TorchScript file containing the GB energy model.");
  params.addRequiredParam<Real>("interface_width", "Coarsest interface width in model.");
  return params;
}

PairwiseAnisotropicGBEnergy::PairwiseAnisotropicGBEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _grad_grain1(getInputBuffer("grad_grain1_buffer")),
    _grad_grain2(getInputBuffer("grad_grain2_buffer")),
    _dsigma_dgrain1(getOutputBufferByName(getParam<TensorOutputBufferName>("dsigma_dgrad_grain1"))),
    _dsigma_dgrain2(getOutputBufferByName(getParam<TensorOutputBufferName>("dsigma_dgrad_grain2"))),
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
PairwiseAnisotropicGBEnergy::computeBuffer()
{
  if (_grad_grain1.dim() < 1 || _grad_grain2.dim() < 1)
    mooseError("grad_grain1_buffer and grad_grain2_buffer must have at least one dimension.");

  if (_grad_grain1.size(-1) != 3 || _grad_grain2.size(-1) != 3)
    mooseError("grad_grain1_buffer and grad_grain2_buffer must have trailing component dimensions "
               "of size 3 each.");

  const auto batch_size = _grad_grain1.numel() / 3;
  std::vector<int64_t> normal_shape(_grad_grain1.sizes().begin(), _grad_grain1.sizes().end());
  std::vector<int64_t> output_shape(_grad_grain1.sizes().begin(), _grad_grain1.sizes().end() - 1);

  // Release previous step's outputs before allocating new ones
  _u = torch::Tensor();
  _dsigma_dgrain1 = torch::Tensor();
  _dsigma_dgrain2 = torch::Tensor();

  // 1. Reshape gradient arrays
  auto grad_grain1_buffer = _grad_grain1.reshape({batch_size, 3});
  auto grad_grain2_buffer = _grad_grain2.reshape({batch_size, 3});

  const auto opts = grad_grain1_buffer.options();

  // 2. Pre-filter to interface points only
  torch::Tensor valid_idx;
  {
    torch::NoGradGuard no_grad;
    auto grad_grain1_mag = torch::sqrt(
        (grad_grain1_buffer * grad_grain1_buffer).sum(/*dim=*/1));
    auto grad_grain2_mag = torch::sqrt(
        (grad_grain2_buffer * grad_grain2_buffer).sum(/*dim=*/1));
    auto valid_mask = (grad_grain1_mag >= _gradient_threshold) &
                      (grad_grain2_mag >= _gradient_threshold);
    // auto delta_grad = grad_grain1_buffer - grad_grain2_buffer;
    // auto delta_grad_mag = torch::sqrt(
    //     (delta_grad * delta_grad).sum(/*dim=*/1));
    // auto valid_mask = delta_grad_mag >= _gradient_threshold;
    valid_idx = torch::where(valid_mask)[0];
  }
  const auto N_interface = valid_idx.size(0);

  auto grad_grain1_valid = grad_grain1_buffer.index_select(0, valid_idx).requires_grad_(true);
  auto grad_grain2_valid = grad_grain2_buffer.index_select(0, valid_idx).requires_grad_(true);

  // deallocate the full grid copy
  grad_grain1_buffer = torch::Tensor();
  grad_grain2_buffer = torch::Tensor();

  // 3. Run model and autograd for valid interface points
  torch::Tensor gamma_valid, dsigma_dgrad_grain1_valid, dsigma_dgrad_grain2_valid;
  if (N_interface > 0)
  {
    // Calculate n = (grad_gr1 - grad_gr2) / |(grad_gr1 - grad_gr2)|
    {
      auto delta_grad_valid = grad_grain1_valid - grad_grain2_valid;
      auto delta_grad_mag_valid =
          torch::sqrt((delta_grad_valid * delta_grad_valid).sum(/*dim=*/1, /*keepdim=*/true));

      auto n_hat = delta_grad_valid / delta_grad_mag_valid;
      // grad_mag_valid and n_hat are released at end of this scope,
      // though the graph retains its own references until grad() is called
      gamma_valid = _surrogate->forward({n_hat}).toTensor().reshape({N_interface});
    }
    auto grads = torch::autograd::grad({gamma_valid.sum()},
                                       {grad_grain1_valid, grad_grain2_valid},
                                       /*grad_outputs=*/{},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/false);

    dsigma_dgrad_grain1_valid = grads[0];
    dsigma_dgrad_grain2_valid = grads[1];

    // Graph is freed by grad(); release remaining live tensors
    grad_grain1_valid = torch::Tensor();
    grad_grain2_valid = torch::Tensor();
    gamma_valid = gamma_valid.detach();
  }

  // 4. Scatter results into full-grid outputs
  auto gamma_full = torch::zeros({batch_size}, opts);
  auto dsigma_dgrad_grain1_full = torch::zeros({batch_size, 3}, opts);
  auto dsigma_dgrad_grain2_full = torch::zeros({batch_size, 3}, opts);

  if (N_interface > 0)
  {
    gamma_full.index_put_({valid_idx}, gamma_valid);
    dsigma_dgrad_grain1_full.index_put_({valid_idx}, dsigma_dgrad_grain1_valid);
    dsigma_dgrad_grain2_full.index_put_({valid_idx}, dsigma_dgrad_grain2_valid);
  }

  _u = gamma_full.reshape(output_shape);
  _dsigma_dgrain1 = dsigma_dgrad_grain1_full.reshape(normal_shape);
  _dsigma_dgrain2 = dsigma_dgrad_grain2_full.reshape(normal_shape);
}