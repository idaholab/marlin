/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>
#include <variant>

/**
 * TraceableSize - A dimension value that can be either concrete or symbolic
 *
 * During JIT tracing, tensor dimensions should be captured as traceable tensor
 * references using torch::jit::tracer::getSizeOf() rather than concrete integers.
 * This allows a single traced graph to work with different tensor sizes at runtime.
 *
 * Following the NEML2 approach for batch-size-independent tracing.
 */
struct TraceableSize : public std::variant<int64_t, torch::Tensor>
{
  using std::variant<int64_t, torch::Tensor>::variant;

  /// Default constructor (creates concrete size of 0)
  TraceableSize() : std::variant<int64_t, torch::Tensor>(int64_t(0)) {}

  /// Construct from concrete size
  TraceableSize(int64_t size) : std::variant<int64_t, torch::Tensor>(size) {}

  /// Construct from traceable tensor (dimension reference)
  TraceableSize(const torch::Tensor & tensor) : std::variant<int64_t, torch::Tensor>(tensor) {}

  /// Check if this size is traceable (i.e., a tensor reference rather than concrete)
  bool isTraceable() const noexcept { return std::holds_alternative<torch::Tensor>(*this); }

  /// Return pointer to traceable tensor if traceable, otherwise nullptr
  const torch::Tensor * traceable() const noexcept
  {
    return std::holds_alternative<torch::Tensor>(*this) ? &std::get<torch::Tensor>(*this) : nullptr;
  }

  /// Return the concrete size value (extracts from tensor if traceable)
  int64_t concrete() const
  {
    if (std::holds_alternative<int64_t>(*this))
      return std::get<int64_t>(*this);
    else
      return std::get<torch::Tensor>(*this).item<int64_t>();
  }

  /// Return size as a tensor (converts concrete to scalar tensor if needed)
  torch::Tensor asTensor() const
  {
    if (std::holds_alternative<torch::Tensor>(*this))
      return std::get<torch::Tensor>(*this);
    else
      return torch::tensor(std::get<int64_t>(*this));
  }

  /// Implicit conversion to int64_t for convenience
  operator int64_t() const { return concrete(); }
};
