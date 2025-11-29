/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TraceableTensorShape.h"

#include <torch/csrc/jit/frontend/tracer.h>

namespace TraceableUtils
{

/**
 * Extract tensor dimensions as traceable sizes.
 *
 * During JIT tracing, this uses torch::jit::tracer::getSizeOf() to create
 * symbolic dimension references in the traced graph. These symbolic references
 * evaluate to actual sizes at runtime, allowing a single trace to work with
 * different tensor sizes.
 *
 * When not tracing, returns concrete integer dimensions.
 *
 * @param tensor The tensor to extract dimensions from
 * @param ndim Number of dimensions to extract (0 = all dimensions)
 * @return TraceableTensorShape containing either symbolic or concrete dimensions
 */
inline TraceableTensorShape
extractTraceableSizes(const torch::Tensor & tensor, int64_t ndim = 0)
{
  const auto actual_ndim = ndim > 0 ? ndim : tensor.dim();

  if (torch::jit::tracer::isTracing())
  {
    // During tracing: create symbolic dimension references
    TraceableTensorShape sizes;
    sizes.reserve(actual_ndim);
    for (int64_t i = 0; i < actual_ndim; ++i)
      sizes.emplace_back(torch::jit::tracer::getSizeOf(tensor, i));
    return sizes;
  }
  else
  {
    // Not tracing: use concrete dimensions
    TraceableTensorShape sizes;
    sizes.reserve(actual_ndim);
    auto tensor_sizes = tensor.sizes();
    for (int64_t i = 0; i < actual_ndim; ++i)
      sizes.emplace_back(tensor_sizes[i]);
    return sizes;
  }
}

/**
 * Get a single dimension as a traceable size.
 *
 * @param tensor The tensor to get the dimension from
 * @param dim The dimension index
 * @return TraceableSize (symbolic during tracing, concrete otherwise)
 */
inline TraceableSize
getTraceableSize(const torch::Tensor & tensor, int64_t dim)
{
  const auto actual_dim = dim >= 0 ? dim : tensor.dim() + dim;

  if (torch::jit::tracer::isTracing())
    return TraceableSize(torch::jit::tracer::getSizeOf(tensor, actual_dim));
  else
    return TraceableSize(tensor.size(actual_dim));
}

} // namespace TraceableUtils
