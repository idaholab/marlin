/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TracedComputeSequence.h"
#include "TensorOperatorBase.h"

#include <memory>
#include <vector>
#include <variant>

class TensorProblem;

/**
 * JITExecutor - Manages JIT tracing and execution for a collection of computes
 *
 * Handles building an execution plan that splits around non-JITable computes,
 * creating TracedComputeSequence objects for contiguous JITable sections.
 */
class JITExecutor
{
public:
  /**
   * Construct a JITExecutor
   * @param problem Reference to the TensorProblem (for buffer access)
   * @param enabled Whether JIT is enabled
   */
  JITExecutor(TensorProblem & problem, bool enabled = true);

  /**
   * Build execution plan from sorted computes, splitting around non-JITable ones.
   * This should be called after dependency resolution.
   * @param sorted_computes Vector of computes in dependency-resolved order
   */
  void buildExecutionPlan(const std::vector<std::shared_ptr<TensorOperatorBase>> & sorted_computes);

  /// Execute all segments (handles traced and non-traced)
  void execute();

  /// Invalidate all cached traces
  void invalidateAllCaches();

  /// Enable/disable JIT (for debugging)
  void setEnabled(bool enabled) { _enabled = enabled; }
  bool isEnabled() const { return _enabled; }

  /// Get the number of traced sequences in the execution plan
  std::size_t getTracedSequenceCount() const;

  /// Get the number of non-JITable computes in the execution plan
  std::size_t getNonJITableCount() const;

private:
  TensorProblem & _problem;
  bool _enabled;

  /// Execution segment: either a traced sequence or a non-JITable compute
  struct ExecutionSegment
  {
    enum class Type
    {
      TRACED_SEQUENCE,
      NON_JITABLE_COMPUTE
    };
    Type type;

    /// For TRACED_SEQUENCE type
    std::unique_ptr<TracedComputeSequence> traced_sequence;

    /// For NON_JITABLE_COMPUTE type
    std::shared_ptr<TensorOperatorBase> non_jitable_compute;
  };

  /// The execution plan: alternating traced sequences and non-JITable computes
  std::vector<ExecutionSegment> _execution_plan;

  /// Original computes list (for fallback non-JIT execution)
  std::vector<std::shared_ptr<TensorOperatorBase>> _all_computes;
};
