/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PairwiseAnisotropicGBEnergy.h"

#include "MarlinUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>

registerMooseObject("MarlinApp", PairwiseAnisotropicGBEnergy);

InputParameters
PairwiseAnisotropicGBEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();

  params.addClassDescription(
      "Evaluates anisotropic grain-boundary energy between a pair of grains from a TorchScript "
      "model, given the gradient buffers of each grain's order parameter.");

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

  params.addRequiredRangeCheckedParam<Real>(
      "interface_width", "interface_width > 0", "Coarsest interface width in model.");

  params.addRangeCheckedParam<unsigned int>(
      "chunk_size",
      65536,
      "chunk_size > 0",
      "Maximum number of spatial points processed at once. Smaller values reduce peak "
      "GPU memory usage at the cost of additional TorchScript launches.");

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
    _interface_width(getParam<Real>("interface_width")),
    _chunk_size(getParam<unsigned int>("chunk_size"))
{
  const auto ref = torch::empty({0}, MooseTensor::floatTensorOptions());
  _surrogate->to(ref.device(), ref.scalar_type(), /* non_blocking = */ false);
  _surrogate->eval();

  // See AnisotropicGBEnergy for the derivation of this threshold from the
  // tanh interface profile.
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

  if (_grad_grain1.sizes() != _grad_grain2.sizes())
    mooseError("grad_grain1_buffer and grad_grain2_buffer must have identical shapes.");

  const int64_t batch_size = _grad_grain1.numel() / 3;

  std::vector<int64_t> normal_shape(_grad_grain1.sizes().begin(), _grad_grain1.sizes().end());
  std::vector<int64_t> output_shape(_grad_grain1.sizes().begin(), _grad_grain1.sizes().end() - 1);

  // Release previous step's outputs before allocating new ones.
  _u = torch::Tensor();
  _dsigma_dgrain1 = torch::Tensor();
  _dsigma_dgrain2 = torch::Tensor();

  // These are views into the original buffers and do not allocate full-grid copies.
  const auto grad_grain1_buffer = _grad_grain1.reshape({batch_size, 3});
  const auto grad_grain2_buffer = _grad_grain2.reshape({batch_size, 3});

  const auto opts = grad_grain1_buffer.options();

  // gamma_full is initialized to NaN. Calculated interface values overwrite these
  // entries chunk-by-chunk. After all chunks have been evaluated, any remaining
  // NaNs are replaced with the largest GBE calculated anywhere in the domain, to
  // avoid an artificial low-energy region in the bulk.
  auto gamma_full = torch::full({batch_size}, std::numeric_limits<float>::quiet_NaN(), opts);

  // Derivatives remain zero outside the evaluated interface region: there is no
  // torque contribution where there is no interface.
  auto dsigma_dgrad_grain1_full = torch::zeros({batch_size, 3}, opts);
  auto dsigma_dgrad_grain2_full = torch::zeros({batch_size, 3}, opts);

  // Largest GBE encountered so far, kept on-device to avoid a CPU/GPU sync.
  // Undefined until the first valid interface point is evaluated.
  torch::Tensor gamma_max;

  const Real gradient_threshold_sq = _gradient_threshold * _gradient_threshold;

  // Process the entire domain in chunks. Both interface filtering and
  // surrogate/autograd evaluation occur within each chunk, so peak memory
  // usage is bounded by chunk_size rather than the full grid.
  for (int64_t chunk_start = 0; chunk_start < batch_size;
       chunk_start += static_cast<int64_t>(_chunk_size))
  {
    const int64_t chunk_length =
        std::min<int64_t>(static_cast<int64_t>(_chunk_size), batch_size - chunk_start);

    // Views into the input buffers; these do not allocate copies.
    auto grad_grain1_chunk = grad_grain1_buffer.narrow(/*dim=*/0, chunk_start, chunk_length);
    auto grad_grain2_chunk = grad_grain2_buffer.narrow(/*dim=*/0, chunk_start, chunk_length);

    torch::Tensor valid_idx;

    // Find interface points within this chunk. A point is only treated as
    // interface if all three of: grain 1's gradient, grain 2's gradient, and
    // their difference are resolvable, i.e. large enough relative to the
    // gradient threshold. The third condition matters at triple junctions,
    // where two order parameters can rise together against a third grain
    // and their difference (the quantity normalized below) nearly vanishes
    // even though each gradient individually clears the threshold.
    //
    // Squared magnitudes are compared instead of taking sqrt(), since this
    // gives the same threshold condition while avoiding extra square roots.
    {
      torch::NoGradGuard no_grad;

      auto grad_grain1_mag_sq = (grad_grain1_chunk * grad_grain1_chunk).sum(/*dim=*/1);
      auto grad_grain2_mag_sq = (grad_grain2_chunk * grad_grain2_chunk).sum(/*dim=*/1);
      auto delta_grad_chunk = grad_grain1_chunk - grad_grain2_chunk;
      auto delta_grad_mag_sq = (delta_grad_chunk * delta_grad_chunk).sum(/*dim=*/1);

      auto valid_mask = (grad_grain1_mag_sq >= gradient_threshold_sq) &
                        (grad_grain2_mag_sq >= gradient_threshold_sq) &
                        (delta_grad_mag_sq >= gradient_threshold_sq);

      valid_idx = torch::where(valid_mask)[0];
    }

    const int64_t N_interface = valid_idx.size(0);

    if (N_interface == 0)
      continue;

    // Gather only the valid points from this chunk. detach() ensures these
    // become independent autograd leaves; the full input buffers never
    // participate in the autograd graph.
    auto grad_grain1_valid =
        grad_grain1_chunk.index_select(0, valid_idx).detach().requires_grad_(true);
    auto grad_grain2_valid =
        grad_grain2_chunk.index_select(0, valid_idx).detach().requires_grad_(true);

    // Compute the interface normal n = (grad_gr1 - grad_gr2) / |grad_gr1 - grad_gr2|
    // and evaluate the surrogate model.
    torch::Tensor gamma_valid;
    {
      auto delta_grad_valid = grad_grain1_valid - grad_grain2_valid;
      auto delta_grad_mag_valid =
          torch::sqrt((delta_grad_valid * delta_grad_valid).sum(/*dim=*/1, /*keepdim=*/true));

      auto n_hat = delta_grad_valid / delta_grad_mag_valid;
      // delta_grad_mag_valid and n_hat are released at the end of this scope,
      // though the graph retains its own references until grad() is called.
      gamma_valid = _surrogate->forward({n_hat}).toTensor().reshape({N_interface});
    }

    // Update the running maximum measured GBE, kept as a tensor on the same
    // device so that no CPU/GPU synchronization is required.
    {
      torch::NoGradGuard no_grad;

      auto chunk_max = gamma_valid.detach().max();
      gamma_max = gamma_max.defined() ? torch::maximum(gamma_max, chunk_max) : chunk_max;
    }

    // Differentiate gamma with respect to each grain's gradient.
    auto grads = torch::autograd::grad({gamma_valid.sum()},
                                       {grad_grain1_valid, grad_grain2_valid},
                                       /*grad_outputs=*/{},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/false);

    auto dsigma_dgrad_grain1_valid = grads[0];
    auto dsigma_dgrad_grain2_valid = grads[1];

    // Scatter this chunk directly into the final outputs. valid_idx is local
    // to the chunk, so narrow() lets us scatter without constructing a
    // global index array.
    {
      torch::NoGradGuard no_grad;

      auto gamma_output_chunk = gamma_full.narrow(/*dim=*/0, chunk_start, chunk_length);
      auto dsigma_dgrad_grain1_output_chunk =
          dsigma_dgrad_grain1_full.narrow(/*dim=*/0, chunk_start, chunk_length);
      auto dsigma_dgrad_grain2_output_chunk =
          dsigma_dgrad_grain2_full.narrow(/*dim=*/0, chunk_start, chunk_length);

      gamma_output_chunk.index_copy_(/*dim=*/0, valid_idx, gamma_valid.detach());
      dsigma_dgrad_grain1_output_chunk.index_copy_(
          /*dim=*/0, valid_idx, dsigma_dgrad_grain1_valid.detach());
      dsigma_dgrad_grain2_output_chunk.index_copy_(
          /*dim=*/0, valid_idx, dsigma_dgrad_grain2_valid.detach());
    }

    // All tensors associated with this chunk's surrogate/autograd graph go
    // out of scope at the end of the loop iteration; since retain_graph is
    // false, that graph storage can then be reused by the caching allocator.
  }

  // Fill all non-interface locations with the largest GBE actually measured,
  // or 1.0 if no interface point was found anywhere in the domain.
  {
    torch::NoGradGuard no_grad;

    if (gamma_max.defined())
      gamma_full.copy_(torch::where(torch::isnan(gamma_full), gamma_max, gamma_full));
    else
      gamma_full.fill_(1.0);
  }

  _u = gamma_full.reshape(output_shape);
  _dsigma_dgrain1 = dsigma_dgrad_grain1_full.reshape(normal_shape);
  _dsigma_dgrain2 = dsigma_dgrad_grain2_full.reshape(normal_shape);
}
