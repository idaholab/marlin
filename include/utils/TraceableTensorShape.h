/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TraceableSize.h"
#include <vector>

/**
 * TraceableTensorShape - A collection of TraceableSize values representing tensor shape
 *
 * Used to store tensor shapes where each dimension can be either concrete (int64_t)
 * or symbolic (torch::Tensor reference for JIT tracing).
 *
 * Following the NEML2 approach for batch-size-independent tracing.
 */
class TraceableTensorShape : public std::vector<TraceableSize>
{
public:
  using std::vector<TraceableSize>::vector;

  /// Construct from concrete tensor shape
  TraceableTensorShape(c10::IntArrayRef shape)
  {
    reserve(shape.size());
    for (auto s : shape)
      emplace_back(s);
  }

  /// Construct from a tensor by extracting its shape
  explicit TraceableTensorShape(const torch::Tensor & tensor)
  {
    auto sizes = tensor.sizes();
    reserve(sizes.size());
    for (auto s : sizes)
      emplace_back(s);
  }

  /// Get concrete shape (evaluates all traceable dimensions)
  std::vector<int64_t> concrete() const
  {
    std::vector<int64_t> result;
    result.reserve(size());
    for (const auto & s : *this)
      result.push_back(s.concrete());
    return result;
  }

  /// Get shape as IntArrayRef (only works if all dimensions are concrete)
  /// Note: Returns a vector since IntArrayRef requires stable storage
  std::vector<int64_t> toIntVector() const { return concrete(); }

  /// Check if any dimension is traceable
  bool hasTraceableDimensions() const
  {
    for (const auto & s : *this)
      if (s.isTraceable())
        return true;
    return false;
  }

  /// Create a 1D tensor containing all dimensions (traceable or concrete)
  torch::Tensor asTensor() const
  {
    std::vector<torch::Tensor> dim_tensors;
    dim_tensors.reserve(size());
    for (const auto & s : *this)
      dim_tensors.push_back(s.asTensor());
    return torch::stack(dim_tensors);
  }

  /// Slice operation (returns subset of dimensions)
  TraceableTensorShape slice(int64_t start, int64_t end) const
  {
    TraceableTensorShape result;
    auto actual_start = start >= 0 ? start : static_cast<int64_t>(size()) + start;
    auto actual_end = end >= 0 ? end : static_cast<int64_t>(size()) + end;
    for (int64_t i = actual_start; i < actual_end && i < static_cast<int64_t>(size()); ++i)
      result.push_back((*this)[i]);
    return result;
  }

  /// Get first N dimensions
  TraceableTensorShape slice(int64_t n) const { return slice(0, n); }
};
