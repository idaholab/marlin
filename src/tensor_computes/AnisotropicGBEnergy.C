/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AnisotropicGBEnergy.h"

#include "MarlinUtils.h"

#include <algorithm>
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
  params.addRangeCheckedParam<unsigned int>(
      "chunk_size",
      65536,
      "chunk_size > 0",
      "Maximum number of spatial points processed at once. Smaller values reduce peak "
      "GPU memory usage at the cost of additional TorchScript launches.");
  return params;
}

AnisotropicGBEnergy::AnisotropicGBEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _gb_gradient_buffer(getInputBuffer("gb_gradient_buffer")),
    _dsigma_dn(getOutputBufferByName(getParam<TensorOutputBufferName>("dsigma_dn"))),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path))),
    _interface_width(getParam<Real>("interface_width")),
    _chunk_size(getParam<unsigned int>("chunk_size"))
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

  const int64_t batch_size = _gb_gradient_buffer.numel() / 3;

  std::vector<int64_t> normal_shape(_gb_gradient_buffer.sizes().begin(),
                                    _gb_gradient_buffer.sizes().end());
  std::vector<int64_t> output_shape(_gb_gradient_buffer.sizes().begin(),
                                    _gb_gradient_buffer.sizes().end() - 1);

  // Release previous step's outputs before allocating new ones.
  _u = torch::Tensor();
  _dsigma_dn = torch::Tensor();

  // A view into the original buffer; this does not allocate a full-grid copy.
  const auto gb_gradient_buffer = _gb_gradient_buffer.reshape({batch_size, 3});

  const auto opts = gb_gradient_buffer.options();

  // gamma_full is initialized to NaN. Calculated interface values overwrite these
  // entries chunk-by-chunk. After all chunks have been evaluated, any remaining
  // NaNs are replaced with the largest GBE calculated anywhere in the domain, to
  // avoid an artificial low-energy region in the bulk.
  auto gamma_full = torch::full({batch_size}, std::numeric_limits<float>::quiet_NaN(), opts);

  // Derivatives remain zero outside the evaluated interface region: there is no
  // torque contribution where there is no interface.
  auto dsigma_dg_full = torch::zeros({batch_size, 3}, opts);

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

    // A view into the input buffer; this does not allocate a copy.
    auto gb_gradient_chunk = gb_gradient_buffer.narrow(/*dim=*/0, chunk_start, chunk_length);

    torch::Tensor valid_idx;

    // Compute the valid (interface) mask on this chunk only, then pre-filter
    // to interface points before running the surrogate model.
    {
      torch::NoGradGuard no_grad;

      auto grad_mag_sq = (gb_gradient_chunk * gb_gradient_chunk).sum(/*dim=*/1);
      auto valid_mask = (grad_mag_sq >= gradient_threshold_sq);

      valid_idx = torch::where(valid_mask)[0];
    }

    const int64_t N_interface = valid_idx.size(0);

    if (N_interface == 0)
      continue;

    // Gather only the valid points from this chunk. detach() ensures these
    // become independent autograd leaves; the full input buffer never
    // participates in the autograd graph.
    auto gb_gradient_valid =
        gb_gradient_chunk.index_select(0, valid_idx).detach().requires_grad_(true);

    torch::Tensor gamma_valid;
    {
      // Recompute from the leaf so autograd differentiates through normalization.
      auto grad_mag_valid =
          torch::sqrt((gb_gradient_valid * gb_gradient_valid).sum(/*dim=*/1, /*keepdim=*/true));

      auto n_hat = gb_gradient_valid / grad_mag_valid;
      // grad_mag_valid and n_hat are released at the end of this scope, though
      // the graph retains its own references until grad() is called.
      gamma_valid = _surrogate->forward({n_hat}).toTensor().reshape({N_interface});
    }

    // Update the running maximum measured GBE, kept as a tensor on the same
    // device so that no CPU/GPU synchronization is required.
    {
      torch::NoGradGuard no_grad;

      auto chunk_max = gamma_valid.detach().max();
      gamma_max = gamma_max.defined() ? torch::maximum(gamma_max, chunk_max) : chunk_max;
    }

    auto grads = torch::autograd::grad({gamma_valid.sum()},
                                       {gb_gradient_valid},
                                       /*grad_outputs=*/{},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/false);
    auto dsigma_dg_valid = grads[0];

    // Scatter this chunk directly into the final outputs. valid_idx is local
    // to the chunk, so narrow() lets us scatter without constructing a
    // global index array.
    {
      torch::NoGradGuard no_grad;

      auto gamma_output_chunk = gamma_full.narrow(/*dim=*/0, chunk_start, chunk_length);
      auto dsigma_dg_output_chunk = dsigma_dg_full.narrow(/*dim=*/0, chunk_start, chunk_length);

      gamma_output_chunk.index_copy_(/*dim=*/0, valid_idx, gamma_valid.detach());
      dsigma_dg_output_chunk.index_copy_(/*dim=*/0, valid_idx, dsigma_dg_valid.detach());
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
  _dsigma_dn = dsigma_dg_full.reshape(normal_shape);
}
