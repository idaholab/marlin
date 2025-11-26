/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>
#include <vector>

/**
 * TraceSchema - Cache key for traced JIT functions
 *
 * Determines when a cached traced graph can be reused. Traces are keyed by
 * the NUMBER of dimensions in each tensor (not their sizes), allowing a single
 * trace to work across different grid sizes. This enables tracing at a small
 * batch size and running at a larger one.
 *
 * Following the NEML2 approach: torch::jit::tracer::getSizeOf() creates symbolic
 * dimension references in the graph that evaluate to actual sizes at runtime.
 */
struct TraceSchema
{
  /// Number of dimensions for each input tensor (NOT the actual sizes)
  std::vector<int64_t> tensor_ndims;

  /// Dispatch key (determines device: CPU, CUDA, etc.)
  at::DispatchKey dispatch_key;

  bool operator==(const TraceSchema & other) const;
  bool operator<(const TraceSchema & other) const;

  /**
   * Create a schema from input tensors and tensor options.
   * Only captures dimension counts (ndim), not concrete sizes.
   * @param inputs Vector of pointers to input tensors
   * @param options TensorOptions specifying device/dtype
   * @return TraceSchema capturing the dimension structure
   */
  static TraceSchema fromTensors(const std::vector<const torch::Tensor *> & inputs,
                                 const torch::TensorOptions & options);
};
