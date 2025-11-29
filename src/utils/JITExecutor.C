/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "JITExecutor.h"
#include "TensorProblem.h"
#include "MarlinUtils.h"

JITExecutor::JITExecutor(TensorProblem & problem, bool enabled)
  : _problem(problem), _enabled(enabled)
{
}

void
JITExecutor::buildExecutionPlan(
    const std::vector<std::shared_ptr<TensorOperatorBase>> & sorted_computes)
{
  _execution_plan.clear();
  _all_computes = sorted_computes;

  if (!_enabled)
    return;

  std::unique_ptr<TracedComputeSequence> current_sequence;

  for (const auto & compute : sorted_computes)
  {
    if (compute->supportsJIT())
    {
      // JITable compute - add to current sequence
      if (!current_sequence)
        current_sequence = std::make_unique<TracedComputeSequence>();
      current_sequence->addCompute(compute);
    }
    else
    {
      // Non-JITable compute - finalize current sequence and add it
      if (current_sequence && !current_sequence->empty())
      {
        current_sequence->finalize();
        ExecutionSegment segment;
        segment.type = ExecutionSegment::Type::TRACED_SEQUENCE;
        segment.traced_sequence = std::move(current_sequence);
        _execution_plan.push_back(std::move(segment));
        current_sequence.reset();
      }

      // Add the non-JITable compute as its own segment
      ExecutionSegment segment;
      segment.type = ExecutionSegment::Type::NON_JITABLE_COMPUTE;
      segment.non_jitable_compute = compute;
      _execution_plan.push_back(std::move(segment));
    }
  }

  // Add any remaining traced sequence
  if (current_sequence && !current_sequence->empty())
  {
    current_sequence->finalize();
    ExecutionSegment segment;
    segment.type = ExecutionSegment::Type::TRACED_SEQUENCE;
    segment.traced_sequence = std::move(current_sequence);
    _execution_plan.push_back(std::move(segment));
  }
}

void
JITExecutor::execute()
{
  if (!_enabled || _execution_plan.empty())
  {
    // Fallback to direct execution
    for (const auto & compute : _all_computes)
    {
      try
      {
        compute->computeBuffer();
      }
      catch (const std::exception & e)
      {
        compute->mooseError("Exception: ", e.what());
      }
    }
    return;
  }

  // Get tensor options from problem
  const auto options = MooseTensor::floatTensorOptions();

  // Execute each segment
  for (auto & segment : _execution_plan)
  {
    switch (segment.type)
    {
      case ExecutionSegment::Type::TRACED_SEQUENCE:
        segment.traced_sequence->execute(_problem, options);
        break;

      case ExecutionSegment::Type::NON_JITABLE_COMPUTE:
        try
        {
          segment.non_jitable_compute->computeBuffer();
        }
        catch (const std::exception & e)
        {
          segment.non_jitable_compute->mooseError("Exception: ", e.what());
        }
        break;
    }
  }
}

void
JITExecutor::invalidateAllCaches()
{
  for (auto & segment : _execution_plan)
    if (segment.type == ExecutionSegment::Type::TRACED_SEQUENCE)
      segment.traced_sequence->invalidateCache();
}

std::size_t
JITExecutor::getTracedSequenceCount() const
{
  std::size_t count = 0;
  for (const auto & segment : _execution_plan)
    if (segment.type == ExecutionSegment::Type::TRACED_SEQUENCE)
      ++count;
  return count;
}

std::size_t
JITExecutor::getNonJITableCount() const
{
  std::size_t count = 0;
  for (const auto & segment : _execution_plan)
    if (segment.type == ExecutionSegment::Type::NON_JITABLE_COMPUTE)
      ++count;
  return count;
}
