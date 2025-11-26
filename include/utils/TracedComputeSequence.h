/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TraceSchema.h"
#include "TensorOperatorBase.h"

#include <torch/torch.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

class TensorProblem;

/**
 * TracedComputeSequence - A contiguous sequence of JIT-traceable computes
 *
 * Manages a sequence of TensorOperatorBase instances that can be traced together
 * as a single PyTorch JIT graph. Handles tracing, caching, and execution.
 */
class TracedComputeSequence
{
public:
  TracedComputeSequence();

  /// Add a compute to this sequence
  void addCompute(std::shared_ptr<TensorOperatorBase> compute);

  /// Get all computes in this sequence
  const std::vector<std::shared_ptr<TensorOperatorBase>> & getComputes() const { return _computes; }

  /// Get input buffer names (buffers read but not written by this sequence)
  const std::set<std::string> & getInputBufferNames() const { return _input_buffer_names; }

  /// Get output buffer names (buffers written by this sequence)
  const std::set<std::string> & getOutputBufferNames() const { return _output_buffer_names; }

  /// Finalize the sequence after all computes have been added (computes I/O sets)
  void finalize();

  /// Execute this sequence (traced or untraced based on cache state)
  void execute(TensorProblem & problem, const torch::TensorOptions & options);

  /// Force retracing on next execution
  void invalidateCache();

  /// Check if this sequence is empty
  bool empty() const { return _computes.empty(); }

  /// Get the number of computes in this sequence
  std::size_t size() const { return _computes.size(); }

private:
  /// Computes in execution order
  std::vector<std::shared_ptr<TensorOperatorBase>> _computes;

  /// External input buffer names (read but not produced internally)
  std::set<std::string> _input_buffer_names;

  /// Output buffer names (produced by this sequence)
  std::set<std::string> _output_buffer_names;

  /// Ordered list of input buffer names (for consistent stack ordering)
  std::vector<std::string> _ordered_input_names;

  /// Ordered list of output buffer names (for consistent stack ordering)
  std::vector<std::string> _ordered_output_names;

  /// Cached traced graph executors keyed by TraceSchema
  std::map<TraceSchema, torch::jit::GraphExecutor> _traced_executors;

  /// Whether finalize() has been called
  bool _finalized;

  /// Collect input tensors into a JIT stack
  torch::jit::Stack collectInputStack(TensorProblem & problem) const;

  /// Assign outputs from JIT stack back to problem buffers
  void assignOutputStack(torch::jit::Stack & stack, TensorProblem & problem) const;

  /// Assign inputs from JIT stack to problem buffers (used during tracing)
  void assignInputStack(const torch::jit::Stack & stack, TensorProblem & problem) const;

  /// Collect outputs from problem buffers into a JIT stack (used during tracing)
  torch::jit::Stack collectOutputStack(TensorProblem & problem) const;

  /// Perform tracing of the compute sequence
  void trace(TensorProblem & problem, const TraceSchema & schema, const torch::TensorOptions & options);

  /// Lookup buffer name by tensor pointer (for JIT variable naming)
  std::string lookupBufferName(const torch::Tensor & tensor, TensorProblem & problem) const;
};
