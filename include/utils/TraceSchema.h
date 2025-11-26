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
 * Determines when a cached traced graph can be reused. Different batch
 * dimensions or devices require different traced graphs.
 */
struct TraceSchema
{
  /// Batch dimensions of all input buffers involved in the trace
  std::vector<int64_t> batch_dims;

  /// Dispatch key (determines device: CPU, CUDA, etc.)
  at::DispatchKey dispatch_key;

  bool operator==(const TraceSchema & other) const;
  bool operator<(const TraceSchema & other) const;

  /**
   * Create a schema from input tensors and tensor options.
   * @param inputs Vector of pointers to input tensors
   * @param options TensorOptions specifying device/dtype
   * @return TraceSchema capturing the current input configuration
   */
  static TraceSchema fromTensors(const std::vector<const torch::Tensor *> & inputs,
                                 const torch::TensorOptions & options);
};
